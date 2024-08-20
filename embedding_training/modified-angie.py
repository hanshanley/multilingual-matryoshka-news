"""
Purpose: This script processes a dataset of news article pairs to create embeddings using the AngIE loss.
It includes data processing, dataset augmentation, custom sampling, and training of a multilingual embedding model using 
cosine and angle loss functions. The script implements a training loop with validation and model checkpointing to identify 
similar news articles across different levels of granularity.

See the Results Section for additional details about training different baseline models using this method for comparison 
against matryoshka trained embeddings. 
"""

# Import necessary libraries
import pandas as pd
import torch
import numpy as np
import json
import csv
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from torch import optim
from datasets import load_dataset
import re
from tqdm import tqdm
import random
import gc
from collections import defaultdict

# Set device to GPU if available
torch.cuda.is_available()
device = torch.cuda.current_device()
torch.cuda.empty_cache()  # Clear GPU cache
gc.collect()  # Garbage collection to free memory

# Constants
BATCH_SIZE = 16
EMBEDDING_SIZE = 768

# Step 1: Process the "NEWS PAIR DATASET" file to create a dictionary of URL pairs
f = open('NEWS PAIR DATASET')
url_to_also_appear = dict()
for line in f:
    try:
        line = json.loads(line)
        url1 = line[0]
        url2 = line[2]
        if line[-1] > 0:
            if url1 not in url_to_also_appear:
                url_to_also_appear[url1] = set()
            url_to_also_appear[url1].add(url2)
            url_to_also_appear[url1].add(url1)
            if url2 not in url_to_also_appear:
                url_to_also_appear[url2] = set()
            url_to_also_appear[url2].add(url1)
            url_to_also_appear[url2].add(url2)
        else:
            if url1 not in url_to_also_appear:
                url_to_also_appear[url1] = set()
            url_to_also_appear[url1].add(url1)
            if url2 not in url_to_also_appear:
                url_to_also_appear[url2] = set()
            url_to_also_appear[url2].add(url2)
    except Exception as e:
        print(e)

# Expand URL pairs by including related URLs
for url in url_to_also_appear:
    for other_url in list(url_to_also_appear[url]):
        url_to_also_appear[url].update(url_to_also_appear[other_url])

# Step 2: Create label dictionaries
label_to_urls = dict()
url_to_label = dict()
current_label = -1
f = open('NEWS PAIR DATASET')
for line in f:
    try:
        line = json.loads(line)
        url1 = line[0]
        url2 = line[2]
        if line[-1] > 0:
            if url1 not in url_to_label and url2 not in url_to_label:
                current_label += 1
                url_to_label[url1] = current_label
                for url in url_to_also_appear[url1]:
                    url_to_label[url] = current_label
                url_to_label[url2] = current_label
                for url in url_to_also_appear[url2]:
                    url_to_label[url] = current_label
            elif url1 not in url_to_label and url2 in url_to_label:
                url2label = url_to_label[url2]
                url_to_label[url1] = url2label
                for url in url_to_also_appear[url1]:
                    url_to_label[url] = url2label
            elif url1 in url_to_label and url2 not in url_to_label:
                url1label = url_to_label[url1]
                url_to_label[url2] = url1label
                for url in url_to_also_appear[url2]:
                    url_to_label[url] = url1label
        else:
            if url1 not in url_to_label and url2 not in url_to_label:
                current_label += 1
                url_to_label[url1] = current_label
                for url in url_to_also_appear[url1]:
                    url_to_label[url] = current_label
                current_label += 1
                url_to_label[url2] = current_label
                for url in url_to_also_appear[url2]:
                    url_to_label[url] = current_label
            elif url1 not in url_to_label and url2 in url_to_label:
                current_label += 1
                url_to_label[url1] = current_label
                for url in url_to_also_appear[url1]:
                    url_to_label[url] = current_label
            elif url1 in url_to_label and url2 not in url_to_label:
                current_label += 1
                url_to_label[url2] = current_label
                for url in url_to_also_appear[url2]:
                    url_to_label[url] = current_label
    except Exception as e:
        print(e)

# Step 3: Create label-to-URL dictionary
label_to_urls = dict()
for url in url_to_label:
    label = url_to_label[url]
    if label not in label_to_urls:
        label_to_urls[label] = set()
    label_to_urls[label].add(url)

# Step 4: Load dataset lines and labels
f = open('NEWS PAIR DATASET')
lines = []
num_bad = 0
for line in f:
    try:
        line = json.loads(line)
        lines.append([line[1], line[3], line[-1]])
    except Exception as e:
        print(e)

# Function to load embedding dataset
def load_embedding_dataset(file_name):
    dataset = []
    labels = []
    with open(file_name, 'r') as fp:
        for line in fp:
            try:
                line = json.loads(line)
                dataset.append([line[1], line[3], line[-1]])
                labels.append([url_to_label[line[0]], url_to_label[line[2]]])
            except Exception as e:
                print(e)
    return dataset, labels

# Load the dataset
dataset, labels = load_embedding_dataset('NEWS PAIR DATASET')

# Shuffle and split the dataset into training and validation sets
c = list(zip(dataset, labels))
random.shuffle(c)
dataset, labels = zip(*c)

train_dataset = dataset[:int(0.90 * len(dataset))]
val_dataset = dataset[int(0.90 * len(dataset)):]
train_labels = labels[:int(0.90 * len(labels))]
val_labels = labels[int(0.90 * len(labels)):]

# Define a custom sampler to create batches without repeating labels
class NonRepeatingBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.index_to_labels = defaultdict(list)
        
        for idx, label in enumerate(labels):
            self.index_to_labels[idx].append(label[0])
            self.index_to_labels[idx].append(label[1])
            
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        label_indices = list(self.index_to_labels.keys())
        random.shuffle(label_indices)
        current_batch = []
        current_labels = set()
        for idx in label_indices:
            labels = self.index_to_labels[idx]
            if labels[0] not in current_labels and labels[1] not in current_labels:
                current_batch.append(idx)
                current_labels.add(labels[0])
                current_labels.add(labels[1])
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_labels = set()
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

# Instantiate samplers for training and validation data
train_sampler = NonRepeatingBatchSampler(train_labels, BATCH_SIZE)
val_sampler = NonRepeatingBatchSampler(val_labels, BATCH_SIZE)

# Define a custom Dataset class for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        text_pair1 = ["query: " + str(x[0]) for x in data]
        text_pair2 = ["query: " + str(x[1]) for x in data]
        margins = [float(x[2]) for x in data]

        text_pair1_encoding = self.tokenizer(text_pair1, return_tensors='pt', max_length=512, padding=True, truncation=True)
        text_pair2_encoding = self.tokenizer(text_pair2, return_tensors='pt', max_length=512, padding=True, truncation=True)

        text_pair1_token_ids = torch.LongTensor(text_pair1_encoding['input_ids'])
        text_pair1_attention_mask = torch.LongTensor(text_pair1_encoding['attention_mask'])
        text_pair2_token_ids = torch.LongTensor(text_pair2_encoding['input_ids'])
        text_pair2_attention_mask = torch.LongTensor(text_pair2_encoding['attention_mask'])

        return (text_pair1_token_ids, text_pair1_attention_mask,
                text_pair2_token_ids, text_pair2_attention_mask, margins)

    def collate_fn(self, all_data):
        (text_pair1_token_ids, text_pair1_attention_mask,
         text_pair2_token_ids, text_pair2_attention_mask, margins) = self.pad_data(all_data)

        batched_data = {
                'text_pair1_token_ids': text_pair1_token_ids,
                'text_pair1_attention_mask': text_pair1_attention_mask,
                'text_pair2_token_ids': text_pair2_token_ids,
                'text_pair2_attention_mask': text_pair2_attention_mask,
                'margins': margins
            }

        return batched_data

# Initialize training and validation datasets and dataloaders
args = Object()
args.tokenizer = 'intfloat/multilingual-e5-base'

train_data = EmbeddingDataset(train_dataset, args)
train_data_dataloader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=train_data.collate_fn)

val_data = EmbeddingDataset(val_dataset, args)
val_data_dataloader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=val_data.collate_fn)

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir='cache', use_fast=False)
model = AutoModel.from_pretrained(args.tokenizer, cache_dir='cache')
model = model.to(device)
model = model.train()

# Mean Pooling function to average token embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Custom cosine loss function. Adapted from: https://github.com/SeanLee97/AnglE
def cosine_loss(y_true, y_pred, tau = 20.0) -> torch.Tensor:
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

# Custom angle loss function. Adapted from https://github.com/SeanLee97/AnglE
def angle_loss(y_true, y_pred, tau=1.0, pooling_strategy='sum'):
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    if pooling_strategy == 'sum':
        pooling = torch.sum(y_pred, dim=1)
    elif pooling_strategy == 'mean':
        pooling = torch.mean(y_pred, dim=1)
    else:
        raise ValueError(f'Unsupported pooling strategy: {pooling_strategy}')
    y_pred = torch.abs(pooling) * tau  # absolute delta angle
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

# Function to calculate loss. 
def calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm, b_margins):
    temp = 0.05
    data_full = torch.cat((embeddings_1_data_norm, embeddings_2_data_norm), dim=0)
    data_full_diff = torch.cat((embeddings_1_diff_data_norm, embeddings_2_diff_data_norm), dim=0)
    cosine_matrix = torch.mm(data_full, data_full_diff.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices + len(embeddings_1_data_norm)] = True
    mask[indices + len(embeddings_1_data_norm), indices] = True
    mask[indices + len(embeddings_1_data_norm), indices + len(embeddings_1_data_norm)] = True
    
    top = torch.exp(cosine_matrix / temp) * mask.to(device)
    top = torch.sum(top, dim=1)
    bottom = torch.exp(cosine_matrix / temp)
    bottom = torch.sum(bottom, dim=1)
    loss = torch.sum(-torch.log(top / bottom))
    return loss

# Function to interleave rows for cosine angle loss
def interleave_rows(tensor1, tensor2, embedding_size):
    stacked = torch.stack((tensor1, tensor2), dim=0)
    permuted = stacked.permute(1, 0, 2)
    interleaved = permuted.reshape(-1, embedding_size)
    return interleaved

# Function to calculate cosine angle loss
def calculate_cosine_angle_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm, b_margins, embedding_size):
    combined_one = interleave_rows(embeddings_1_data_norm, embeddings_1_diff_data_norm, embedding_size)
    combined_two = interleave_rows(embeddings_2_data_norm, embeddings_2_diff_data_norm, embedding_size)
    combined_pair = interleave_rows(embeddings_1_data_norm, embeddings_2_data_norm, embedding_size)
    combined_pair_diff = interleave_rows(embeddings_1_diff_data_norm, embeddings_2_diff_data_norm, embedding_size)
    combined_all = torch.cat((combined_one, combined_two, combined_pair, combined_pair_diff), dim=0)
    total_b_margins = torch.cat((torch.ones(embeddings_1_data_norm.size(0)), torch.ones(embeddings_1_data_norm.size(0)), torch.tensor(b_margins), torch.tensor(b_margins)), dim=0)
    return cosine_loss(total_b_margins.to(device), combined_all) + angle_loss(total_b_margins.to(device), combined_all) + calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm, b_margins)

# Optimizer setup
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)



def model_eval(dataloader, model, device):
    """
    Evaluate the model on the provided dataloader.
    
    Args:
    - dataloader (DataLoader): The data loader containing validation data.
    - model (nn.Module): The model to evaluate.
    - device (torch.device): The device to run the model on (CPU or GPU).
    
    Returns:
    - total_loss (float): The total loss over the validation dataset.
    """
    
    model.eval()  # Switch the model to evaluation mode (disables dropout, etc.)
    
    with torch.no_grad():  # Disable gradient computation
        total_loss = 0 
        
        # Iterate over batches in the dataloader
        for batch in tqdm(dataloader, desc='Evaluation in progress'):
            # Extract batch data and move to the specified device
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_margins = (
                batch['text_pair1_token_ids'],
                batch['text_pair1_attention_mask'], 
                batch['text_pair2_token_ids'], 
                batch['text_pair2_attention_mask'],
                batch['margins']
            )
            
            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_margins = torch.tensor(b_margins).to(device)

            # Get embeddings for the first and second text pairs
            embeddings_1 = model(b_ids_1, b_mask_1)
            embeddings_1 = mean_pooling(embeddings_1, b_mask_1)
        
            embeddings_1_diff = model(b_ids_1, b_mask_1)
            embeddings_1_diff = mean_pooling(embeddings_1_diff, b_mask_1)
        
            embeddings_2 = model(b_ids_2, b_mask_2)
            embeddings_2 = mean_pooling(embeddings_2, b_mask_2)
        
            embeddings_2_diff = model(b_ids_2, b_mask_2)
            embeddings_2_diff = mean_pooling(embeddings_2_diff, b_mask_2)

            # Compute third embeddings (Matryoshka approach)
            embeddings_1_third = embeddings_1[:, :int(EMBEDDING_SIZE/1)]
            embeddings_1_diff_third  = embeddings_1_diff[:, :int(EMBEDDING_SIZE/1)]
            embeddings_2_third  = embeddings_2[:, :int(EMBEDDING_SIZE/1)]
            embeddings_2_diff_third = embeddings_2_diff[:, :int(EMBEDDING_SIZE/1)]
            
            # Normalize embeddings
            embeddings_1_norms_third = embeddings_1_third.norm(dim=1, keepdim=True)
            embeddings_1_data_norm_third = embeddings_1_third / embeddings_1_norms_third
            
            embeddings_1_diff_norms_third = embeddings_1_diff_third.norm(dim=1, keepdim=True)
            embeddings_1_diff_data_norm_third = embeddings_1_diff_third / embeddings_1_diff_norms_third
            
            embeddings_2_norms_third = embeddings_2_third.norm(dim=1, keepdim=True)
            embeddings_2_data_norm_third = embeddings_2_third / embeddings_2_norms_third
            
            embeddings_2_diff_norms_third = embeddings_2_diff_third.norm(dim=1, keepdim=True)
            embeddings_2_diff_data_norm_third = embeddings_2_diff_third / embeddings_2_diff_norms_third

            # Adjust margins for high-margin cases
            indices_high = (b_margins == 0.75).nonzero(as_tuple=True)[0]
            b_margins_third = b_margins.clone()
            b_margins_third[indices_high] = 1
    
            # Calculate the loss using cosine angle difference
            loss = calculate_cosine_angle_loss(
                embeddings_1_data_norm_third, 
                embeddings_1_diff_data_norm_third, 
                embeddings_2_data_norm_third, 
                embeddings_2_diff_data_norm_third,
                b_margins_third,
                int(EMBEDDING_SIZE/1)
            ) / BATCH_SIZE
        
            total_loss += float(loss.item())  # Accumulate the loss
    
        return total_loss  # Return the total loss over the dataset



# Directory setup for saving models
import os
if not os.path.isdir('FOLDER'):
    os.mkdir('FOLDER')

import torch
from tqdm import tqdm

# Initialize epoch count and set a very high initial minimum loss value
epoch = 0
minimum_loss = float('inf')

# Loop over epochs (up to a very large number, acting as an infinite loop until manually stopped)
for epoch in range(0, 1000000):
    num_batches = 0  # Counter for the number of batches processed
    model.train()  # Set the model to training mode

    # Iterate over batches in the training dataloader
    for batch in tqdm(train_data_dataloader, desc=f'train-{epoch}'):
        # Extract batch data and move to the specified device (e.g., GPU)
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_margins = (
            batch['text_pair1_token_ids'],
            batch['text_pair1_attention_mask'], 
            batch['text_pair2_token_ids'], 
            batch['text_pair2_attention_mask'], 
            batch['margins']
        )
        
        # Reset the gradients of the optimizer
        opt.zero_grad()

        # Move the data to the specified device
        b_ids_1, b_mask_1 = b_ids_1.to(device), b_mask_1.to(device)
        b_ids_2, b_mask_2 = b_ids_2.to(device), b_mask_2.to(device)
        b_margins = torch.tensor(b_margins).to(device)

        # Get embeddings for the first and second text pairs
        embeddings_1 = model(b_ids_1, b_mask_1)
        embeddings_1 = mean_pooling(embeddings_1, b_mask_1)
    
        embeddings_1_diff = model(b_ids_1, b_mask_1)
        embeddings_1_diff = mean_pooling(embeddings_1_diff, b_mask_1)
    
        embeddings_2 = model(b_ids_2, b_mask_2)
        embeddings_2 = mean_pooling(embeddings_2, b_mask_2)
    
        embeddings_2_diff = model(b_ids_2, b_mask_2)
        embeddings_2_diff = mean_pooling(embeddings_2_diff, b_mask_2)

        # Compute third embeddings (Matryoshka approach)
        embeddings_1_third = embeddings_1[:, :int(EMBEDDING_SIZE / 1)]
        embeddings_1_diff_third = embeddings_1_diff[:, :int(EMBEDDING_SIZE / 1)]
        embeddings_2_third = embeddings_2[:, :int(EMBEDDING_SIZE / 1)]
        embeddings_2_diff_third = embeddings_2_diff[:, :int(EMBEDDING_SIZE / 1)]
        
        # Normalize the embeddings
        embeddings_1_norms_third = embeddings_1_third.norm(dim=1, keepdim=True)
        embeddings_1_data_norm_third = embeddings_1_third / embeddings_1_norms_third
        
        embeddings_1_diff_norms_third = embeddings_1_diff_third.norm(dim=1, keepdim=True)
        embeddings_1_diff_data_norm_third = embeddings_1_diff_third / embeddings_1_diff_norms_third
        
        embeddings_2_norms_third = embeddings_2_third.norm(dim=1, keepdim=True)
        embeddings_2_data_norm_third = embeddings_2_third / embeddings_2_norms_third
        
        embeddings_2_diff_norms_third = embeddings_2_diff_third.norm(dim=1, keepdim=True)
        embeddings_2_diff_data_norm_third = embeddings_2_diff_third / embeddings_2_diff_norms_third

        # Adjust margins for high-margin cases
        indices_high = (b_margins == 0.75).nonzero(as_tuple=True)[0]
        b_margins_third = b_margins.clone()
        b_margins_third[indices_high] = 1

        # Calculate the loss using cosine angle difference and normalize by batch size
        loss = calculate_cosine_angle_loss(
            embeddings_1_data_norm_third, 
            embeddings_1_diff_data_norm_third, 
            embeddings_2_data_norm_third, 
            embeddings_2_diff_data_norm_third,
            b_margins_third,
            int(EMBEDDING_SIZE / 1)
        ) / BATCH_SIZE
        
        num_batches += 1  # Increment the batch counter

        # Backpropagation and optimizer step
        loss.backward()
        opt.step()

        # Periodically evaluate the model on validation data
        if num_batches % 1000 == 0:
            validation_loss = model_eval(val_data_dataloader, model, device)
            if validation_loss < minimum_loss:
                # Save the model if the validation loss decreases
                try:
                    torch.save(
                        model.state_dict(), 
                        f'non-multilingual-matryoshka-xlm-roberta-base-calculate_cosine_angle_loss-fixed-regular/20240808-non-multilingual-matryoshka-mpnet-calculate_cosine_angle_loss-base{epoch}-{num_batches}.pt'
                    )
                except Exception as e:
                    print(e)
                minimum_loss = validation_loss  # Update the minimum loss

        # Log the training loss
        try:
            with open('loss.txt', 'a+') as f:
                f.write(str(loss.item()) + "\n")
        except Exception as e:
            print(e)
    
    # Evaluate the model on the validation set and save if it's the best one
    try:
        validation_loss = model_eval(val_data_dataloader, model, device)
        if validation_loss < minimum_loss:
            try:
                torch.save(model.state_dict(), f'FOLDER/model-{epoch}-{num_batches}.pt')
            except Exception as e:
                print(e)
            minimum_loss = validation_loss  # Update minimum loss
    except Exception as e:
        print(e)
