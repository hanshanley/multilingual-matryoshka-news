#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import numpy as np
import json
# Generate a random vector of size 768
import csv

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch import optim
from datasets import load_dataset
import json
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# In[2]:


BATCH_SIZE = 16
EMBEDDING_SIZE= 768



train_dataset = []
train_lines_to_remove = []
f = open('/mnt/projects/qanon_proj/Matryoshka/all_processed_datasets_w_entities_replaced-20241125.jsonl','r')
ind = 0 
for line in f:
    try:
        train_dataset.append(json.loads(line))

    except Exception as e:
        print(e)
    if ind %100000 ==0:
        print(ind)
    ind+=1
f.close()

val_dataset = []
f = open('/mnt/projects/qanon_proj/Matryoshka/validation_multilingual_final-20241008.jsonl','r')
ind = 0 
val_lines_to_remove = []

for line in f:
    try:
        val_dataset.append(json.loads(line))
        
    except Exception as e:
        print(e)
    if ind %100000 ==0:
        print(ind)
    ind+=1
    
f.close()



'''         
def remove_items_at_indices(lst, indices):
    indices_set = set(indices)
    lst[:] = [item for idx, item in enumerate(lst) if idx not in indices_set]
    return lst

train_labels = remove_items_at_indices(train_labels,train_lines_to_remove)

val_labels = remove_items_at_indices(val_labels,val_lines_to_remove)
'''     

from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import random
import torch
from collections import defaultdict
import random
from torch.utils.data import Sampler

from collections import defaultdict
import random
from torch.utils.data import Sampler

class NonRepeatingBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.index_to_labels = defaultdict(list)

        # Organize labels
        for idx, label in enumerate(labels):
            if isinstance(label, (tuple, list)) and len(label) >= 2:
                self.index_to_labels[idx].extend(label[:2])  # Store the first two labels
            else:
                raise ValueError("Expected each label to have at least two elements.")
        
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        label_indices = list(self.index_to_labels.keys())
        random.shuffle(label_indices)

        current_batch = []
        current_labels = set()
        leftovers = []

        for idx in label_indices:
            labels = self.index_to_labels[idx]

            # Check if adding this index will repeat any label
            if labels[0] not in current_labels and labels[1] not in current_labels:
                current_batch.append(idx)
                current_labels.update(labels)

                # If we've reached the desired batch size, store the batch
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_labels = set()
            else:
                # If it can't be added to the current batch, add to leftovers
                leftovers.append(idx)

        # Try to use leftovers to create additional batches
        max_attempts = len(leftovers)  # To avoid infinite loop if no valid batches can be formed
        attempts = 0
        while leftovers and attempts < max_attempts:
            idx = leftovers.pop(0)
            labels = self.index_to_labels[idx]

            # Try to add leftover indices to the current batch if possible
            if labels[0] not in current_labels and labels[1] not in current_labels:
                current_batch.append(idx)
                current_labels.update(labels)
            else:
                # If it still can't be added, put it back to the end of leftovers
                leftovers.append(idx)

            # If we've reached the desired batch size, store the batch
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_labels = set()
            
            attempts += 1

        # Handle any remaining items that couldn't form a full batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
# In[15]:


#train_sampler = NonRepeatingBatchSampler(train_labels, BATCH_SIZE)
#val_sampler = NonRepeatingBatchSampler(val_labels, BATCH_SIZE)


# In[16]:


class EmbeddingDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained(args.tokenizer)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        text_pair1 = ["query: "+str(x[0]) for x in data]
        text_pair2 = ["query: "+str(x[1]) for x in data]
        margins = [float(x[2]) for x in data]

        text_pair1_encoding = self.tokenizer(text_pair1, return_tensors='pt', max_length=512, padding=True, truncation=True)
        text_pair2_encoding = self.tokenizer(text_pair2, return_tensors='pt',  max_length=512,  padding=True, truncation=True)

        text_pair1_token_ids = torch.LongTensor(text_pair1_encoding['input_ids'])
        text_pair1_attention_mask = torch.LongTensor(text_pair1_encoding['attention_mask'])

        text_pair2_token_ids = torch.LongTensor(text_pair2_encoding['input_ids'])
        text_pair2_attention_mask = torch.LongTensor(text_pair2_encoding['attention_mask'])


        return (text_pair1_token_ids, text_pair1_attention_mask,
                text_pair2_token_ids, text_pair2_attention_mask, margins)

    def collate_fn(self, all_data):
        (text_pair1_token_ids,  text_pair1_attention_mask,
         text_pair2_token_ids, text_pair2_attention_mask, margins) = self.pad_data(all_data)

        batched_data = {
                'text_pair1_token_ids': text_pair1_token_ids,
                'text_pair1_attention_mask': text_pair1_attention_mask,
                'text_pair2_token_ids': text_pair2_token_ids,
                'text_pair2_attention_mask': text_pair2_attention_mask,
                'margins': margins
            }

        return batched_data


# In[17]:


import random
from transformers import AutoModel, AutoTokenizer
class Object(object):
    pass
args = Object()
model_name ='intfloat/multilingual-e5-base'#'intfloat/e5-base-v2'#'intfloat/multilingual-e5-base' #'microsoft/mdeberta-v3-base'#'google/umt5-base'
args.tokenizer ='intfloat/multilingual-e5-base'#'intfloat/e5-base-v2'#intfloat/multilingual-e5-base'#microsoft/mdeberta-v3-base'# 'google/umt5-base'
#random.shuffle(dataset)


# In[18]:


train_data = EmbeddingDataset(train_dataset, args)
train_data_dataloader = DataLoader(train_data, shuffle=True,batch_size = BATCH_SIZE,
                                      collate_fn=train_data.collate_fn)


# In[19]:


#train_data = EmbeddingDataset(train_dataset, args)
val_data = EmbeddingDataset(val_dataset, args)
val_data_dataloader = DataLoader(val_data,shuffle=True,batch_size = BATCH_SIZE,
                                      collate_fn=val_data.collate_fn)


# In[20]:
import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir= 'cache',use_fast =False)
model = AutoModel.from_pretrained(model_name, cache_dir= 'cache')

model = model.to(device)
model = model.train()

# In[21]:




# In[22]:


from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# In[23]:


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """  # NOQA
    # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent20241130#L79
    #y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


# In[24]:


def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0, pooling_strategy: str = 'sum'):
    """
    Compute angle loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 1.0

    :return: torch.Tensor, loss value
    """  # NOQA
    #y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()

    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # (a+bi) / (c+di)
    # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
    # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
    # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
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


def calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm, b_margins):
    temp = 0.05
    data_full = torch.cat((embeddings_1_data_norm, embeddings_2_diff_data_norm), dim=0)
    data_full_diff = torch.cat((embeddings_1_diff_data_norm, embeddings_2_data_norm), dim=0)
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


def mse_class_loss(embeddings_1, embeddings_2, labels):
    """
    Computes the Mean Squared Error (MSE) loss between pairs of embeddings 
    that have a label of 1.

    Args:
        embeddings_1 (torch.Tensor): A tensor of shape [batch_size, embedding_dim] 
                                     representing the first set of embeddings.
        embeddings_2 (torch.Tensor): A tensor of shape [batch_size, embedding_dim] 
                                     representing the second set of embeddings.
        labels (torch.Tensor): A tensor of shape [batch_size] containing binary labels 
                               (0 or 1) where 1 indicates that the corresponding pair 
                               of embeddings should be similar.

    Returns:
        torch.Tensor: The MSE loss computed only over the pairs with label 1.
                      If no pairs have label 1, returns a tensor with value 0.
    """
    # Create a mask for the pairs with label 1
    positive_mask = (labels == 1)
    
    # If there are no positive pairs, return zero loss
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings_1.device)
    
    # Compute the MSE loss only for the positive pairs
    loss = F.mse_loss(embeddings_1[positive_mask], embeddings_2[positive_mask])
    return loss





'''
import torch
def calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm,b_margins):
    temp =0.05
    data_full = torch.cat((embeddings_1_data_norm, embeddings_2_diff_data_norm), dim=0)
 
    data_full_diff = torch.cat((embeddings_1_diff_data_norm, embeddings_2_data_norm), dim=0)
    
    cosine_matrix1 = torch.mm(data_full, data_full_diff.t())
    indices = torch.arange(data_full.size(0))
    mask = torch.zeros(cosine_matrix1.size(), dtype=torch.bool)
    mask[indices, indices] = True
    new_weight = torch.zeros(cosine_matrix1.size(), dtype=torch.float)
    indices = torch.arange(embeddings_1_data_norm.size(0))
    new_weight[indices, indices] = 1
    new_weight = torch.exp(new_weight)/torch.sum(torch.exp(new_weight),dim = 1)
    new_weight = torch.sum(new_weight, dim = 1)
    
    top= torch.exp(cosine_matrix1/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix1/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss = torch.sum(-torch.log(top/bottom))
    #print(loss)
    cosine_matrix2 = torch.mm(embeddings_2_data_norm, embeddings_2_diff_data_norm.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix2.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True
    top= torch.exp(cosine_matrix2/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix2/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss+= torch.sum(-torch.log(top/bottom))
    #print(loss)
    
    
    
    cosine_matrix3 = torch.mm(embeddings_1_data_norm, embeddings_2_data_norm.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix3.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True
    
    top= torch.exp(cosine_matrix3/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix3/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss+=  torch.sum(-torch.log(top/bottom))
    
    
    cosine_matrix4 = torch.mm(embeddings_1_data_norm, embeddings_2_diff_data_norm.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix4.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True
    
    top= torch.exp(cosine_matrix4/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix4/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss+= torch.sum(-torch.log(top/bottom))
    
    
    cosine_matrix5 = torch.mm(embeddings_1_diff_data_norm, embeddings_2_diff_data_norm.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix5.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True

    
    top= torch.exp(cosine_matrix5/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix5/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss+=  torch.sum(-torch.log(top/bottom))
    
    
    cosine_matrix6 = torch.mm(embeddings_1_diff_data_norm, embeddings_2_data_norm.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix6.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True

    top= torch.exp(cosine_matrix6/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix6/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss+=  torch.sum(-torch.log(top/bottom))
    return loss
'''

# In[26]:


import torch

def interleave_rows(tensor1, tensor2, embedding_size):
    # Stack the tensors along a new dimension (results in shape (2, 32, 768))
    stacked = torch.stack((tensor1, tensor2), dim=0)
    
    # Permute to swap the first two dimensions (results in shape (32, 2, 768))
    permuted = stacked.permute(1, 0, 2)
    
    # Reshape to flatten the first two dimensions (results in shape (64, 768))
    interleaved = permuted.reshape(-1, embedding_size)
    
    return interleaved




import torch
def calculate_cosine_angle_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm,b_margins, embedding_size):
    combined_one = interleave_rows(embeddings_1_data_norm,embeddings_1_diff_data_norm,embedding_size)
    combined_two = interleave_rows(embeddings_2_data_norm,embeddings_2_diff_data_norm,embedding_size)
    
    combined_pair = interleave_rows(embeddings_1_data_norm,embeddings_2_data_norm,embedding_size)
    combined_pair_diff = interleave_rows(embeddings_1_diff_data_norm,embeddings_2_diff_data_norm,embedding_size)

    
    combined_pair2 = interleave_rows(embeddings_1_data_norm,embeddings_2_diff_data_norm,embedding_size)
    combined_pair3 = interleave_rows(embeddings_1_diff_data_norm,embeddings_2_data_norm,embedding_size)

    
    
    combined_all = torch.cat((combined_one,combined_two,combined_pair,combined_pair_diff,combined_pair2,combined_pair3), dim = 0)
    total_b_margins = torch.cat((torch.ones(embeddings_1_data_norm.size(0)),torch.ones(embeddings_1_data_norm.size(0)),torch.tensor(b_margins),torch.tensor(b_margins),torch.tensor(b_margins),torch.tensor(b_margins)),dim = 0 )
    return cosine_loss(total_b_margins.to(device),combined_all)+ angle_loss(total_b_margins.to(device),combined_all)+calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm,b_margins)

# In[28]:


opt = torch.optim.AdamW(model.parameters(), lr=2e-5)


# In[29]:



# In[30]:


def model_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    with torch.no_grad():
        total_loss = 0 
        for batch in tqdm(val_data_dataloader, desc=f'train-{epoch}'):
            b_ids_1, b_mask_1,b_ids_2, b_mask_2, b_margins = (batch['text_pair1_token_ids'],
                                   batch['text_pair1_attention_mask'], batch['text_pair2_token_ids'], batch['text_pair2_attention_mask'],batch['margins'])
    
            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
        
            
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_margins = torch.tensor(b_margins)
            embeddings_1 = model(b_ids_1,b_mask_1)
            embeddings_1 = mean_pooling(embeddings_1, b_mask_1)
        
            embeddings_1_diff = model(b_ids_1,b_mask_1)
            embeddings_1_diff = mean_pooling(embeddings_1_diff, b_mask_1)
        
            embeddings_2 = model(b_ids_2,b_mask_2)
            embeddings_2 = mean_pooling(embeddings_2, b_mask_2)
        
            embeddings_2_diff = model(b_ids_2,b_mask_2)
            embeddings_2_diff = mean_pooling(embeddings_2_diff, b_mask_2)

            indices_high = (b_margins >= 0.25).nonzero(as_tuple=True)[0]
            indices_low = (b_margins < 0.25).nonzero(as_tuple=True)[0]
            b_margins_first = b_margins.clone()
            b_margins_first[indices_high] = 1#b_margins[indices_high]
            b_margins_first[indices_low] = 0
            indices_high = (b_margins >= 0.50).nonzero(as_tuple=True)[0]
            indices_low = (b_margins < 0.50).nonzero(as_tuple=True)[0]
            b_margins_second = b_margins.clone()
            b_margins_second[indices_high] = 1#b_margins[indices_high]
            b_margins_second[indices_low] = 0
            indices_high = (b_margins >= 0.75).nonzero(as_tuple=True)[0]
            indices_low = (b_margins < 0.75).nonzero(as_tuple=True)[0]
            b_margins_third = b_margins.clone()
            b_margins_third[indices_high] = 1#b_margins[indices_high]
            b_margins_third[indices_low] = 0
            #indices_high = (b_margins >= 1.0).nonzero(as_tuple=True)[0]
            #indices_low = (b_margins < 1.0).nonzero(as_tuple=True)[0]
            #b_margins_fourth= b_margins.clone()
            #b_margins_fourth[indices_high] = 1
            #b_margins_fourth[indices_low] = 0
        
            ### FIRST EMBEDDING Matryoshka
            embeddings_1_first = embeddings_1[:, :int(EMBEDDING_SIZE/4)]
            embeddings_1_diff_first = embeddings_1_diff[:, :int(EMBEDDING_SIZE/4)]
            embeddings_2_first = embeddings_2[:, :int(EMBEDDING_SIZE/4)]
            embeddings_2_diff_first = embeddings_2_diff[:, :int(EMBEDDING_SIZE/4)]
            
            ## Normalize
            embeddings_1_norms_first = embeddings_1_first.norm(dim=1, keepdim=True)
            embeddings_1_data_norm_first = embeddings_1_first / embeddings_1_norms_first
            embeddings_1_diff_norms_first = embeddings_1_diff_first.norm(dim=1, keepdim=True)
            embeddings_1_diff_data_norm_first = embeddings_1_diff_first / embeddings_1_diff_norms_first
            embeddings_2_norms_first = embeddings_2_first.norm(dim=1, keepdim=True)
            embeddings_2_data_norm_first = embeddings_2_first / embeddings_2_norms_first
            embeddings_2_diff_norms_first = embeddings_2_diff_first.norm(dim=1, keepdim=True)
            embeddings_2_diff_data_norm_first = embeddings_2_diff_first / embeddings_2_diff_norms_first
    
            ### SECOND EMBEDDING Matryoshka
            embeddings_1_second = embeddings_1[:, :int(EMBEDDING_SIZE/2)]
            embeddings_1_diff_second  = embeddings_1_diff[:, :int(EMBEDDING_SIZE/2)]
            embeddings_2_second  = embeddings_2[:, :int(EMBEDDING_SIZE/2)]
            embeddings_2_diff_second = embeddings_2_diff[:, :int(EMBEDDING_SIZE/2)]
            
            ## Normalize
            embeddings_1_norms_second = embeddings_1_second.norm(dim=1, keepdim=True)
            embeddings_1_data_norm_second = embeddings_1_second / embeddings_1_norms_second
            embeddings_1_diff_norms_second = embeddings_1_diff_second.norm(dim=1, keepdim=True)
            embeddings_1_diff_data_norm_second = embeddings_1_diff_second / embeddings_1_diff_norms_second
            embeddings_2_norms_second = embeddings_2_second.norm(dim=1, keepdim=True)
            embeddings_2_data_norm_second = embeddings_2_second / embeddings_2_norms_second
            embeddings_2_diff_norms_second = embeddings_2_diff_second.norm(dim=1, keepdim=True)
            embeddings_2_diff_data_norm_second = embeddings_2_diff_second / embeddings_2_diff_norms_second
    
    
            ### THIRD EMBEDDING Matryoshka
            embeddings_1_third = embeddings_1[:, :int(EMBEDDING_SIZE/1)]
            embeddings_1_diff_third  = embeddings_1_diff[:, :int(EMBEDDING_SIZE/1)]
            embeddings_2_third  = embeddings_2[:, :int(EMBEDDING_SIZE/1)]
            embeddings_2_diff_third = embeddings_2_diff[:, :int(EMBEDDING_SIZE/1)]
            
            ## Normalize
            embeddings_1_norms_third = embeddings_1_third.norm(dim=1, keepdim=True)
            embeddings_1_data_norm_third = embeddings_1_third / embeddings_1_norms_third
            embeddings_1_diff_norms_third = embeddings_1_diff_third.norm(dim=1, keepdim=True)
            embeddings_1_diff_data_norm_third = embeddings_1_diff_third / embeddings_1_diff_norms_third
            embeddings_2_norms_third = embeddings_2_third.norm(dim=1, keepdim=True)
            embeddings_2_data_norm_third = embeddings_2_third / embeddings_2_norms_third
            embeddings_2_diff_norms_third = embeddings_2_diff_third.norm(dim=1, keepdim=True)
            embeddings_2_diff_data_norm_third = embeddings_2_diff_third / embeddings_2_diff_norms_third
    
    
            ### FOURTH EMBEDDING Matryoshka
            '''
            embeddings_1_fourth = embeddings_1[:, :EMBEDDING_SIZE]
            embeddings_1_diff_fourth  = embeddings_1_diff[:, :EMBEDDING_SIZE]
            embeddings_2_fourth  = embeddings_2[:, :EMBEDDING_SIZE]
            embeddings_2_diff_fourth = embeddings_2_diff[:, :EMBEDDING_SIZE]
            
            ## Normalize
            embeddings_1_norms_fourth = embeddings_1_fourth.norm(dim=1, keepdim=True)
            embeddings_1_data_norm_fourth = embeddings_1_fourth / embeddings_1_norms_fourth
            embeddings_1_diff_norms_fourth = embeddings_1_diff_fourth.norm(dim=1, keepdim=True)
            embeddings_1_diff_data_norm_fourth = embeddings_1_diff_fourth / embeddings_1_diff_norms_fourth
            embeddings_2_norms_fourth = embeddings_2_fourth.norm(dim=1, keepdim=True)
            embeddings_2_data_norm_fourth = embeddings_2_fourth / embeddings_2_norms_fourth
            embeddings_2_diff_norms_fourth = embeddings_2_diff_fourth.norm(dim=1, keepdim=True)
            embeddings_2_diff_data_norm_fourth = embeddings_2_diff_fourth / embeddings_2_diff_norms_fourth
            '''
    
            loss = calculate_cosine_angle_loss(embeddings_1_data_norm_first, embeddings_1_diff_data_norm_first, embeddings_2_data_norm_first, embeddings_2_diff_data_norm_first,b_margins_first,int(EMBEDDING_SIZE/4))/BATCH_SIZE
            loss += calculate_cosine_angle_loss(embeddings_1_data_norm_second, embeddings_1_diff_data_norm_second, embeddings_2_data_norm_second, embeddings_2_diff_data_norm_second,b_margins_second,int(EMBEDDING_SIZE/2))/BATCH_SIZE
            loss += calculate_cosine_angle_loss(embeddings_1_data_norm_third, embeddings_1_diff_data_norm_third, embeddings_2_data_norm_third, embeddings_2_diff_data_norm_third,b_margins_third,int(EMBEDDING_SIZE/1))/BATCH_SIZE
            #loss += calculate_cosine_angle_loss(embeddings_1_data_norm_fourth, embeddings_1_diff_data_norm_fourth, embeddings_2_data_norm_fourth, embeddings_2_diff_data_norm_fourth,b_margins_fourth,int(EMBEDDING_SIZE))/BATCH_SIZE
            total_loss+=float(loss.item())
    
        return total_loss # f1, #prec, recall, report, y_pred, y_true


# In[31]:


import os
if not os.path.isdir('/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205'):
    os.mkdir('/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205')
epoch = 0 

#model.load_state_dict(torch.load('multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205/'+'multilingual-20250205-matryoshka-e5-calculate_cosine_angle_loss-base0-160000.pt'),strict=False)
#checkpoint = torch.load('/mnt/projects/qanon_proj/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205/opt-state0-160000') 

          
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
#opt.load_state_dict(checkpoint['optimizer_state_dict'])


minimum_loss = 999999999999999
for epoch in range(0,1000000):
    num_batches = 0 
    #if epoch == 0:
    #    num_batches = 160000
        
    model = model.train()
    for batch in tqdm(train_data_dataloader, desc=f'train-{epoch}'):
        #model = model.train()
        b_ids_1, b_mask_1,b_ids_2, b_mask_2, b_margins = (batch['text_pair1_token_ids'],
                                   batch['text_pair1_attention_mask'], batch['text_pair2_token_ids'], batch['text_pair2_attention_mask'],batch['margins'])
        opt.zero_grad()
        #print(batch)
        b_ids_1 = b_ids_1.to(device)
        b_mask_1 = b_mask_1.to(device)
    
        
        b_ids_2 = b_ids_2.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_margins = torch.tensor(b_margins)
        embeddings_1 = model(b_ids_1,b_mask_1)
        embeddings_1 = mean_pooling(embeddings_1, b_mask_1)
    
        embeddings_1_diff = model(b_ids_1,b_mask_1)
        embeddings_1_diff = mean_pooling(embeddings_1_diff, b_mask_1)
    
        embeddings_2 = model(b_ids_2,b_mask_2)
        embeddings_2 = mean_pooling(embeddings_2, b_mask_2)
    
        embeddings_2_diff = model(b_ids_2,b_mask_2)
        embeddings_2_diff = mean_pooling(embeddings_2_diff, b_mask_2)

        indices_high = (b_margins >= 0.25).nonzero(as_tuple=True)[0]
        indices_low = (b_margins < 0.25).nonzero(as_tuple=True)[0]
        b_margins_first = b_margins.clone()
        b_margins_first[indices_high] = 1#b_margins[indices_high]
        b_margins_first[indices_low] = 0
        indices_high = (b_margins >= 0.50).nonzero(as_tuple=True)[0]
        indices_low = (b_margins < 0.50).nonzero(as_tuple=True)[0]
        b_margins_second = b_margins.clone()
        b_margins_second[indices_high] = 1#b_margins[indices_high]
        b_margins_second[indices_low] = 0
        indices_high = (b_margins >= 0.75).nonzero(as_tuple=True)[0]
        indices_low = (b_margins < 0.75).nonzero(as_tuple=True)[0]
        b_margins_third = b_margins.clone()
        b_margins_third[indices_high] = 1#b_margins[indices_high]
        b_margins_third[indices_low] = 0
        #indices_high = (b_margins >= 1.0).nonzero(as_tuple=True)[0]
        #indices_low = (b_margins < 1.0).nonzero(as_tuple=True)[0]
        #b_margins_fourth= b_margins.clone()
        #b_margins_fourth[indices_high] = 1
        #b_margins_fourth[indices_low] = 0
    
        ### FIRST EMBEDDING Matryoshka
        embeddings_1_first = embeddings_1[:, :int(EMBEDDING_SIZE/4)]
        embeddings_1_diff_first = embeddings_1_diff[:, :int(EMBEDDING_SIZE/4)]
        embeddings_2_first = embeddings_2[:, :int(EMBEDDING_SIZE/4)]
        embeddings_2_diff_first = embeddings_2_diff[:, :int(EMBEDDING_SIZE/4)]
        
        ## Normalize
        embeddings_1_norms_first = embeddings_1_first.norm(dim=1, keepdim=True)
        embeddings_1_data_norm_first = embeddings_1_first / embeddings_1_norms_first
        embeddings_1_diff_norms_first = embeddings_1_diff_first.norm(dim=1, keepdim=True)
        embeddings_1_diff_data_norm_first = embeddings_1_diff_first / embeddings_1_diff_norms_first
        embeddings_2_norms_first = embeddings_2_first.norm(dim=1, keepdim=True)
        embeddings_2_data_norm_first = embeddings_2_first / embeddings_2_norms_first
        embeddings_2_diff_norms_first = embeddings_2_diff_first.norm(dim=1, keepdim=True)
        embeddings_2_diff_data_norm_first = embeddings_2_diff_first / embeddings_2_diff_norms_first

        ### SECOND EMBEDDING Matryoshka
        embeddings_1_second = embeddings_1[:, :int(EMBEDDING_SIZE/2)]
        embeddings_1_diff_second  = embeddings_1_diff[:, :int(EMBEDDING_SIZE/2)]
        embeddings_2_second  = embeddings_2[:, :int(EMBEDDING_SIZE/2)]
        embeddings_2_diff_second = embeddings_2_diff[:, :int(EMBEDDING_SIZE/2)]
        
        ## Normalize
        embeddings_1_norms_second = embeddings_1_second.norm(dim=1, keepdim=True)
        embeddings_1_data_norm_second = embeddings_1_second / embeddings_1_norms_second
        embeddings_1_diff_norms_second = embeddings_1_diff_second.norm(dim=1, keepdim=True)
        embeddings_1_diff_data_norm_second = embeddings_1_diff_second / embeddings_1_diff_norms_second
        embeddings_2_norms_second = embeddings_2_second.norm(dim=1, keepdim=True)
        embeddings_2_data_norm_second = embeddings_2_second / embeddings_2_norms_second
        embeddings_2_diff_norms_second = embeddings_2_diff_second.norm(dim=1, keepdim=True)
        embeddings_2_diff_data_norm_second = embeddings_2_diff_second / embeddings_2_diff_norms_second


        ### THIRD EMBEDDING Matryoshka
        embeddings_1_third = embeddings_1[:, :int(EMBEDDING_SIZE/1)]
        embeddings_1_diff_third  = embeddings_1_diff[:, :int(EMBEDDING_SIZE/1)]
        embeddings_2_third  = embeddings_2[:, :int(EMBEDDING_SIZE/1)]
        embeddings_2_diff_third = embeddings_2_diff[:, :int(EMBEDDING_SIZE/1)]
        
        ## Normalize
        embeddings_1_norms_third = embeddings_1_third.norm(dim=1, keepdim=True)
        embeddings_1_data_norm_third = embeddings_1_third / embeddings_1_norms_third
        embeddings_1_diff_norms_third = embeddings_1_diff_third.norm(dim=1, keepdim=True)
        embeddings_1_diff_data_norm_third = embeddings_1_diff_third / embeddings_1_diff_norms_third
        embeddings_2_norms_third = embeddings_2_third.norm(dim=1, keepdim=True)
        embeddings_2_data_norm_third = embeddings_2_third / embeddings_2_norms_third
        embeddings_2_diff_norms_third = embeddings_2_diff_third.norm(dim=1, keepdim=True)
        embeddings_2_diff_data_norm_third = embeddings_2_diff_third / embeddings_2_diff_norms_third


        ### FOURTH EMBEDDING Matryoshka
        '''
        embeddings_1_fourth = embeddings_1[:, :EMBEDDING_SIZE]
        embeddings_1_diff_fourth  = embeddings_1_diff[:, :EMBEDDING_SIZE]
        embeddings_2_fourth  = embeddings_2[:, :EMBEDDING_SIZE]
        embeddings_2_diff_fourth = embeddings_2_diff[:, :EMBEDDING_SIZE]
        
        ## Normalize
        embeddings_1_norms_fourth = embeddings_1_fourth.norm(dim=1, keepdim=True)
        embeddings_1_data_norm_fourth = embeddings_1_fourth / embeddings_1_norms_fourth
        embeddings_1_diff_norms_fourth = embeddings_1_diff_fourth.norm(dim=1, keepdim=True)
        embeddings_1_diff_data_norm_fourth = embeddings_1_diff_fourth / embeddings_1_diff_norms_fourth
        embeddings_2_norms_fourth = embeddings_2_fourth.norm(dim=1, keepdim=True)
        embeddings_2_data_norm_fourth = embeddings_2_fourth / embeddings_2_norms_fourth
        embeddings_2_diff_norms_fourth = embeddings_2_diff_fourth.norm(dim=1, keepdim=True)
        embeddings_2_diff_data_norm_fourth = embeddings_2_diff_fourth / embeddings_2_diff_norms_fourth
        '''

        loss = calculate_cosine_angle_loss(embeddings_1_data_norm_first, embeddings_1_diff_data_norm_first, embeddings_2_data_norm_first, embeddings_2_diff_data_norm_first,b_margins_first,int(EMBEDDING_SIZE/4))/BATCH_SIZE
        loss += calculate_cosine_angle_loss(embeddings_1_data_norm_second, embeddings_1_diff_data_norm_second, embeddings_2_data_norm_second, embeddings_2_diff_data_norm_second,b_margins_second,int(EMBEDDING_SIZE/2))/BATCH_SIZE
        loss += calculate_cosine_angle_loss(embeddings_1_data_norm_third, embeddings_1_diff_data_norm_third, embeddings_2_data_norm_third, embeddings_2_diff_data_norm_third,b_margins_third,int(EMBEDDING_SIZE/1))/BATCH_SIZE
        #loss += calculate_cosine_angle_loss(embeddings_1_data_norm_fourth, embeddings_1_diff_data_norm_fourth, embeddings_2_data_norm_fourth, embeddings_2_diff_data_norm_fourth,b_margins_fourth,int(EMBEDDING_SIZE))/BATCH_SIZE
        
        
        

        loss.backward()
        opt.step()
        num_batches+=1
        if num_batches %10000 ==0:
            validation_loss = model_eval(val_data_dataloader,model, device) 
            if validation_loss < minimum_loss:
                try:
                    torch.save(model.state_dict(), '/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205/multilingual-20250205-matryoshka-e5-calculate_cosine_angle_loss-base'+str(epoch)+'-'+str(num_batches)+'.pt')
                    torch.save({
                    'epoch': epoch,
                    'num_batches': num_batches,
                    'optimizer_state_dict': opt.state_dict(),
                    }, '/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205/opt-state'+str(epoch)+'-'+str(num_batches))
                except Exception as e:
                    print(e)
                minium_loss = validation_loss
                model = model.train()
            model = model.train()
        #if num_batches % 250 ==0:
        
        try:
            f = open('/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-scne-loss-train_loss-fixed-20250205.txt','a+')
            f.write(str(loss.item())+"\n")
            f.close()
        except Exception as e:
            print(e)
        print(loss.item())
    print('HERE')
    try:
        torch.save(model.state_dict(), '/mnt/projects/controversyworld/Matryoshka/multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular-20250205/multilingual-20250205-matryoshka-e5-base'+str(epoch)+'.pt')
    except Exception as e:
        print(e)





