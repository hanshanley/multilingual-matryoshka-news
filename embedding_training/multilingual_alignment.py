#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import numpy as np
import json
import csv
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch import optim
from datasets import load_dataset
from tqdm import tqdm
import langdetect
from bs4 import BeautifulSoup
import re
import os
import justext
import unicodedata

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

# Function to clean HTML and JavaScript from text
def clean_html_js(text):
    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    # Remove inline JavaScript
    text = re.sub(r'function\s*\w*\(.*?\)\s*{[^}]*}', '', text, flags=re.DOTALL)
    text = re.sub(r'var\s*\w*\s*=\s*[^;]*;', '', text, flags=re.DOTALL)
    text = re.sub(r'setInterval\([^)]*\)', '', text, flags=re.DOTALL)
    # Remove include virtual
    text = re.sub(r'include\s*virtual="[^"]*"', '', text, flags=re.DOTALL)
    # Remove conditional comments
    text = re.sub(r'\[if\s*!IE\][^\[]*\[endif\]', '', text, flags=re.DOTALL)
    # Remove remaining unwanted characters
    cleaned_text = re.sub(r'\\n', ' ', text)
    cleaned_text = re.sub(r'\\t', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to process and normalize the text
def process(article):
    normalized_string = unicodedata.normalize('NFKD', article)
    normalized_string = normalized_string.replace('’','\'').replace('‘','\'').replace('“','\"').replace('”','\"')
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', normalized_string)
    cleaned_text = re.sub('(http)(.+?)(?:\s)', ' ', text)
    cleaned_text = re.sub('<.*?>', '', cleaned_text) 
    cleaned_text = cleaned_text.replace('\t',' ').replace('\\t',' ').replace('\n','').replace('\\n','').replace('\r','').replace('\\r','')
    return clean_html_js(remove_extraneous_javascript(cleaned_text.strip()))

# Function to remove extraneous JavaScript from text
def remove_extraneous_javascript(text):
    patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Match <script>...</script> blocks
        r'window\..*?;',  # Match window.<something>;
        r'addEventListener\([^\)]*\)',  # Match addEventListener(...)
        r'function\s*\(.*?\)\s*{.*?}',  # Match function definitions
        r'\bif\b\s*\(.*?\)\s*{.*?}',  # Match if conditions
        r'\bvar\b\s+[^\n;]+;',  # Match var declarations
        r'\bconst\b\s+[^\n;]+;',  # Match const declarations
        r'\blet\b\s+[^\n;]+;',  # Match let declarations
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Remove remaining unwanted characters
    text = re.sub(r'\n\s*\n', '\n', text)
    cleaned_text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'{.*?}', '', cleaned_text, flags=re.DOTALL)
    return text.strip()

# Load data from JSON files
with open('/mnt/projects/qanon_proj/topic-diff/huati-chayi2/url_to_english_text-20240811.json') as f:
    url_to_english_text = json.load(f)

# Load data from JSONL file and filter based on url_to_english_text keys
with open('/mnt/projects/qanon_proj/topic-diff/huati-chayi2/multilingual_finetune_paragraph1_paragraph2_label-20240812.jsonl') as f:
    all_lines = []
    num_seen = 0
    for line in f:
        try:
            line = json.loads(line)
            if line[0] in url_to_english_text and line[2] in url_to_english_text:
                all_lines.append([line[1], line[3], url_to_english_text[line[0]], url_to_english_text[line[2]], line[-1]])
                num_seen += 1
                if num_seen % 100000 == 0:
                    print(num_seen)
        except Exception as e:
            print(e)

# Shuffle the lines for training
random.shuffle(all_lines)
training_lines = all_lines

# Model and tokenizer names
teacher_model_name = 'intfloat/multilingual-e5-base'
student_model_name = 'intfloat/multilingual-e5-base'

# Load teacher model and tokenizer
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, cache_dir='cache', use_fast=False)
teacher_model = AutoModel.from_pretrained(teacher_model_name, cache_dir='cache')
teacher_model.load_state_dict(torch.load('/mnt/projects/qanon_proj/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular/multilingual-20240808-matryoshka-e5-calculate_cosine_angle_loss-base1-1000.pt'))
teacher_model = teacher_model.eval()

# Load student model and tokenizer
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, cache_dir='cache', use_fast=False)
student_model = AutoModel.from_pretrained(student_model_name, cache_dir='cache')
student_model.load_state_dict(torch.load('/mnt/projects/qanon_proj/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular/multilingual-20240808-matryoshka-e5-calculate_cosine_angle_loss-base1-1000.pt'))
student_model = student_model.train()

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

# Move models to device (GPU or CPU)
student_model = student_model.to(device)
teacher_model = teacher_model.to(device)

# Define mean pooling function for embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Dataset class for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        text_pair1 = ["query: " + str(x[0]) for x in data]
        text_pair2 = ["query: " + str(x[1]) for x in data]
        english_text_pair3 = ["query: " + str(x[2]) for x in data]
        english_text_pair4 = ["query: " + str(x[3]) for x in data]
        margins = [float(x[4]) for x in data]

        text_pair1_encoding = self.student_tokenizer(text_pair1, return_tensors='pt', max_length=512, padding=True, truncation=True)
        text_pair2_encoding = self.student_tokenizer(text_pair2, return_tensors='pt', max_length=512, padding=True, truncation=True)

        text_pair1_token_ids = torch.LongTensor(text_pair1_encoding['input_ids'])
        text_pair1_attention_mask = torch.LongTensor(text_pair1_encoding['attention_mask'])

        text_pair2_token_ids = torch.LongTensor(text_pair2_encoding['input_ids'])
        text_pair2_attention_mask = torch.LongTensor(text_pair2_encoding['attention_mask'])

        text_pair3_encoding = self.teacher_tokenizer(english_text_pair3, return_tensors='pt', max_length=512, padding=True, truncation=True)
        text_pair4_encoding = self.teacher_tokenizer(english_text_pair4, return_tensors='pt', max_length=512, padding=True, truncation=True)

        text_pair3_token_ids = torch.LongTensor(text_pair3_encoding['input_ids'])
        text_pair3_attention_mask = torch.LongTensor(text_pair3_encoding['attention_mask'])

        text_pair4_token_ids = torch.LongTensor(text_pair4_encoding['input_ids'])
        text_pair4_attention_mask = torch.LongTensor(text_pair4_encoding['attention_mask'])

        return (text_pair1_token_ids, text_pair1_attention_mask,
                text_pair2_token_ids, text_pair2_attention_mask,
                text_pair3_token_ids, text_pair3_attention_mask,
                text_pair4_token_ids, text_pair4_attention_mask, margins)

    def collate_fn(self, all_data):
        (text_pair1_token_ids, text_pair1_attention_mask,
         text_pair2_token_ids, text_pair2_attention_mask,
         text_pair3_token_ids, text_pair3_attention_mask,
         text_pair4_token_ids, text_pair4_attention_mask, margins) = self.pad_data(all_data)

        batched_data = {
            'text_pair1_token_ids': text_pair1_token_ids,
            'text_pair1_attention_mask': text_pair1_attention_mask,
            'text_pair2_token_ids': text_pair2_token_ids,
            'text_pair2_attention_mask': text_pair2_attention_mask,
            'text_pair3_token_ids': text_pair3_token_ids,
            'text_pair3_attention_mask': text_pair3_attention_mask,
            'text_pair4_token_ids': text_pair4_token_ids,
            'text_pair4_attention_mask': text_pair4_attention_mask,
            'margins': margins
        }
        return batched_data

# Define arguments for tokenizers
class Object(object):
    pass

args = Object()
args.student_tokenizer = 'intfloat/multilingual-e5-base'
args.teacher_tokenizer = 'intfloat/multilingual-e5-base'

# Batch size for training
BATCH_SIZE = 24

# Split data into training and validation sets
train_dataset = training_lines[:int(0.99 * len(training_lines))]
val_dataset = training_lines[int(0.99 * len(training_lines)):]

# Create data loaders for training and validation
train_data = EmbeddingDataset(train_dataset, args)
train_data_dataloader = DataLoader(train_data, collate_fn=train_data.collate_fn, shuffle=True, batch_size=BATCH_SIZE)

val_data = EmbeddingDataset(val_dataset, args)
val_data_dataloader = DataLoader(val_data, collate_fn=val_data.collate_fn, shuffle=True, batch_size=BATCH_SIZE)

# Define optimizer for training
opt = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

# Evaluation function for the model
def model_eval(dataloader, student_model, teacher_model, device):
    student_model.eval()  # switch to eval mode to turn off randomness like dropout
    total_loss = 0 
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'train-{epoch}'):
            # Move batch data to device
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_ids_3, b_mask_3, b_ids_4, b_mask_4, b_margins = (
                batch['text_pair1_token_ids'], batch['text_pair1_attention_mask'],
                batch['text_pair2_token_ids'], batch['text_pair2_attention_mask'],
                batch['text_pair3_token_ids'], batch['text_pair3_attention_mask'], 
                batch['text_pair4_token_ids'], batch['text_pair4_attention_mask'], 
                batch['margins']
            )
            b_ids_1, b_mask_1, b_ids_2, b_mask_2 = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device)
            b_ids_3, b_mask_3, b_ids_4, b_mask_4 = b_ids_3.to(device), b_mask_3.to(device), b_ids_4.to(device), b_mask_4.to(device)
            
            # Get student and teacher embeddings
            embeddings_1 = mean_pooling(student_model(b_ids_1, b_mask_1), b_mask_1)
            embeddings_2 = mean_pooling(student_model(b_ids_2, b_mask_2), b_mask_2)
            teacher_embeddings_1 = mean_pooling(teacher_model(b_ids_3, b_mask_3), b_mask_3)
            teacher_embeddings_2 = mean_pooling(teacher_model(b_ids_4, b_mask_4), b_mask_4)

            # Compute losses for different embedding sizes
            for i, factor in enumerate([4, 2, 1], 1):
                embeddings_1_student = F.normalize(embeddings_1[:, :int(EMBEDDING_SIZE/factor)], p=2, dim=1)
                embeddings_1_teacher = F.normalize(teacher_embeddings_1[:, :int(EMBEDDING_SIZE/factor)], p=2, dim=1)
                embeddings_2_student = F.normalize(embeddings_2[:, :int(EMBEDDING_SIZE/factor)], p=2, dim=1)
                embeddings_2_teacher = F.normalize(teacher_embeddings_2[:, :int(EMBEDDING_SIZE/factor)], p=2, dim=1)
                
                loss = mse_loss_fn(embeddings_1_student, embeddings_1_teacher)
                loss += mse_loss_fn(embeddings_2_student, embeddings_2_teacher)
                cosine_similarity_student = torch.sum(embeddings_1_student * embeddings_2_student, dim=1)
                cosine_similarity_teacher = torch.sum(embeddings_1_teacher * embeddings_2_teacher, dim=1)
                loss += mse_loss_fn(cosine_similarity_student, cosine_similarity_teacher)
                total_loss += float(loss.item())
        
        return total_loss



# Create a directory to save the model if it doesn't exist
if not os.path.isdir('FOLDER'):
    os.mkdir('FOLDER')

# Initialize variables
minimum_loss = float('inf')  # Start with a very high value for minimum loss
EMBEDDING_SIZE = 768  # Size of the embeddings
mse_loss_fn = nn.MSELoss()  # Mean Squared Error loss function
accumulation_steps = 4  # Number of steps for gradient accumulation

# Training loop
for epoch in range(0, 99999999):  # Infinite loop to train the model
    num_batches = 0  # Initialize batch counter
    student_model = student_model.train()  # Set the model to training mode
    
    # Loop over batches
    for batch in tqdm(train_data_dataloader, desc=f'train-{epoch}'):
        # Extract input data from batch
        b_ids_1, b_mask_1, b_ids_3, b_mask_3, b_margins = (
            batch['text_pair1_token_ids'], batch['text_pair1_attention_mask'],
            batch['text_pair3_token_ids'], batch['text_pair3_attention_mask'],
            batch['margins']
        )

        # Move data to GPU (or CPU if GPU is not available)
        b_ids_1, b_mask_1 = b_ids_1.to(device), b_mask_1.to(device)
        b_ids_3, b_mask_3 = b_ids_3.to(device), b_mask_3.to(device)
        b_margins = torch.tensor(b_margins).to(device)

        # Forward pass: Get the embeddings from the student and teacher models
        embeddings_1 = student_model(b_ids_1, b_mask_1)
        embeddings_1 = mean_pooling(embeddings_1, b_mask_1)
        teacher_embeddings_1 = teacher_model(b_ids_3, b_mask_3)
        teacher_embeddings_1 = mean_pooling(teacher_embeddings_1, b_mask_3)

        # Calculate losses at different embedding sizes
        # 1st embedding size (quarter of the full size)
        embeddings_1_student_1 = F.normalize(embeddings_1[:, :int(EMBEDDING_SIZE / 4)], p=2, dim=1)
        embeddings_1_teacher_1 = F.normalize(teacher_embeddings_1[:, :int(EMBEDDING_SIZE / 4)], p=2, dim=1)
        cosine_similarity_student_1 = torch.sum(embeddings_1_teacher_1 * embeddings_1_teacher_1, dim=1)
        loss_1 = mse_loss_fn(cosine_similarity_student_1, b_margins)

        # 2nd embedding size (half of the full size)
        embeddings_1_student_2 = F.normalize(embeddings_1[:, :int(EMBEDDING_SIZE / 2)], p=2, dim=1)
        embeddings_1_teacher_2 = F.normalize(teacher_embeddings_1[:, :int(EMBEDDING_SIZE / 2)], p=2, dim=1)
        cosine_similarity_student_2 = torch.sum(embeddings_1_student_2 * embeddings_1_teacher_2, dim=1)
        loss_2 = mse_loss_fn(cosine_similarity_student_2, b_margins)

        # Full embedding size
        embeddings_1_student_3 = F.normalize(embeddings_1[:, :int(EMBEDDING_SIZE / 1)], p=2, dim=1)
        embeddings_1_teacher_3 = F.normalize(teacher_embeddings_1[:, :int(EMBEDDING_SIZE / 1)], p=2, dim=1)
        cosine_similarity_student_3 = torch.sum(embeddings_1_student_3 * embeddings_1_teacher_3, dim=1)
        loss_3 = mse_loss_fn(cosine_similarity_student_3, b_margins)

        # Aggregate losses and normalize by batch size
        loss = (loss_1 + loss_2 + loss_3) / BATCH_SIZE

        # Log the loss to a file
        with open('fixed-multilingual_alignment_loss-20240814.txt', 'a+') as f:
            f.write(str(loss.item()) + "\n")
        print(loss.item())

        num_batches += 1

        # Backward pass: Compute gradients and update model parameters
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    validation_loss = model_eval(val_data_dataloader, student_model, teacher_model, device)
    if validation_loss < minimum_loss:
        torch.save(student_model.state_dict(), f'FOLDER/multilingual_alignment-{epoch}-{num_batches}.pt')# Save the model at the end of each epoch
    
