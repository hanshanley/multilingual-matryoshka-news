import pandas as pd
import torch
import numpy as np
import json
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
import random
import json
import random
import json

# In[20]:
import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()



BATCH_SIZE = 16
EMBEDDING_SIZE= 768



f  = open('NEWS PAIR DATASET')
url_to_also_apear = dict()
for line in f:
    try:
        line = json.loads(line)
        url1 = line[0]
        url2 = line[2]
        if line[-1] > 0:
            if url1 not in url_to_also_apear:
                url_to_also_apear[url1] = set()
            url_to_also_apear[url1].add(url2)
            url_to_also_apear[url1].add(url1)
            if url2 not in url_to_also_apear:
                url_to_also_apear[url2] = set()
            url_to_also_apear[url2].add(url1)
            url_to_also_apear[url2].add(url2)
        else:
            if url1 not in url_to_also_apear:
                url_to_also_apear[url1] = set()
            url_to_also_apear[url1].add(url1)
            if url2 not in url_to_also_apear:
                url_to_also_apear[url2] = set()
            url_to_also_apear[url2].add(url2)
    except Exception as e:
        print(e)
for url in url_to_also_apear:
    for other_url in list(url_to_also_apear[url]):
        url_to_also_apear[url].update(url_to_also_apear[other_url])
for url in url_to_also_apear:
    for other_url in list(url_to_also_apear[url]):
        url_to_also_apear[url].update(url_to_also_apear[other_url])




label_to_urls = dict()
url_to_label = dict()
current_label =  -1
f  = open('NEWS PAIR DATASET')
for line in f:
    try:
        line = json.loads(line)
        url1 = line[0]
        url2 = line[2]
        if line[-1] > 0:
            if url1 not in url_to_label and url2 not in url_to_label:
                current_label+=1
                url_to_label[url1] = current_label
                for url in url_to_also_apear[url1]:
                     url_to_label[url] = current_label
                url_to_label[url2] = current_label
                for url in url_to_also_apear[url2]:
                    url_to_label[url] = current_label
            elif url1 not in url_to_label and url2 in url_to_label:
                url2label = url_to_label[url2]
                url_to_label[url1] = url2label
                for url in url_to_also_apear[url1]:
                     url_to_label[url] = url2label
            elif url1 in url_to_label and url2 not in url_to_label:
                url1label = url_to_label[url1]
                url_to_label[url2] = url1label
                for url in url_to_also_apear[url2]:
                     url_to_label[url] = url1label
            else:
                pass
        else:
            if url1 not in url_to_label and url2 not in url_to_label:
                current_label+=1
                url_to_label[url1] = current_label
                for url in url_to_also_apear[url1]:
                     url_to_label[url] = current_label
                current_label+=1
                url_to_label[url2] = current_label
                for url in url_to_also_apear[url2]:
                     url_to_label[url] = current_label
            elif url1 not in url_to_label and url2 in url_to_label:
                current_label+=1
                url_to_label[url1] = current_label
                for url in url_to_also_apear[url1]:
                     url_to_label[url] = current_label
            elif url1 in url_to_label and url2 not in url_to_label:
                current_label+=1
                url_to_label[url2] = current_label
                for url in url_to_also_apear[url2]:
                     url_to_label[url] = current_label
    except Exception as e:
        print(e)


# In[7]:


label_to_urls = dict()
for url in url_to_label:
    label = url_to_label[url]
    if label not  in label_to_urls:
        label_to_urls[label] = set()
    label_to_urls[label].add(url)


# In[8]:






f  = open('NEWS PAIR DATASET')
lines = []
num_bad = 0 
for line in f:
    try:
        line  = json.loads(line)
            
        lines.append([line[1], line[3], line[-1] ])
    except Exception as e:
        print(e)




import json
def load_emebdding_dataset(file_name):
    dataset = []
    labels = []
    with open(file_name, 'r') as fp:
        for line in fp:
            try:
                line = json.loads(line)
                #if len(line[0]) < 25 or len(line[1]) < 25:
                #    continue
                dataset.append([line[1], line[3], line[-1] ])
                labels.append([url_to_label[line[0]],url_to_label[line[2]]])
            except Exception as e:
                print(e)
    return dataset, labels




dataset, labels = load_emebdding_dataset('NES PAIR DATASET')




import random
c = list(zip(dataset, labels))

random.shuffle(c)

dataset, labels = zip(*c)




train_dataset = dataset[:int(0.90*len(dataset))]
val_dataset = dataset[int(0.90*len(dataset)):]

train_labels = labels[:int(0.90*len(labels))]
val_labels = labels[int(0.90*len(labels)):]


# In[14]:


from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import random
import torch
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
        batches = []
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




train_sampler = NonRepeatingBatchSampler(train_labels, BATCH_SIZE)
val_sampler = NonRepeatingBatchSampler(val_labels, BATCH_SIZE)




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




import random
from transformers import UMT5EncoderModel, AutoTokenizer
class Object(object):
    pass
args = Object()
model_name ='intfloat/multilingual-e5-base'
args.tokenizer ='intfloat/multilingual-e5-base'




train_data = EmbeddingDataset(train_dataset, args)
train_data_dataloader = DataLoader(train_data,batch_sampler=train_sampler,
                                      collate_fn=train_data.collate_fn)




val_data = EmbeddingDataset(val_dataset, args)
val_data_dataloader = DataLoader(val_data,batch_sampler=val_sampler,
                                      collate_fn=val_data.collate_fn)



tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir= 'cache',use_fast =False)
model = AutoModel.from_pretrained(model_name, cache_dir= 'cache')

model = model.to(device)
model = model.train()



from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


### Adapted from https://github.com/SeanLee97/AnglE
def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """  
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)



## Apdated from https://github.com/SeanLee97/AnglE
def angle_loss(y_true, y_pred, tau= 1.0, pooling_strategy = 'sum'):
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


# In[25]:


import torch
def calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm,b_margins):
    temp =0.05
    data_full = torch.cat((embeddings_1_data_norm, embeddings_2_data_norm), dim=0)
 
    data_full_diff = torch.cat((embeddings_1_diff_data_norm, embeddings_2_diff_data_norm), dim=0)
    cosine_matrix = torch.mm(data_full, data_full_diff.t())
    indices = torch.arange(embeddings_1_data_norm.size(0))
    mask = torch.zeros(cosine_matrix.size(), dtype=torch.bool)
    mask[indices, indices] = True
    mask[indices, indices] = True
    mask[indices, indices + len(embeddings_1_data_norm)] = True  # Adjust this offset based on your column count
    mask[indices+ len(embeddings_1_data_norm), indices]  = True
    mask[indices+len(embeddings_1_data_norm), indices+len(embeddings_1_data_norm)]  = True
    
    top= torch.exp(cosine_matrix/temp)*mask.to(device)
    top = torch.sum(top,dim =1)
    bottom = torch.exp(cosine_matrix/temp)
    bottom = torch.sum(bottom,dim = 1)
    loss = torch.sum(-torch.log(top/bottom))
    return loss




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
    combined_two = interleave_rows(embeddings_2_data_norm,embeddings_2_data_norm,embedding_size)
    combined_pair = interleave_rows(embeddings_1_data_norm,embeddings_2_data_norm,embedding_size)
    combined_pair_diff = interleave_rows(embeddings_1_diff_data_norm,embeddings_2_diff_data_norm,embedding_size)
    combined_all = torch.cat((combined_one,combined_two,combined_pair_diff,combined_pair_diff), dim = 0)
    total_b_margins = torch.cat((torch.ones(embeddings_1_data_norm.size(0)),torch.ones(embeddings_1_data_norm.size(0)),torch.tensor(b_margins),torch.tensor(b_margins)),dim = 0 )
    return cosine_loss(total_b_margins.to(device),combined_all)+ angle_loss(total_b_margins.to(device),combined_all)+calculate_loss(embeddings_1_data_norm, embeddings_1_diff_data_norm, embeddings_2_data_norm, embeddings_2_diff_data_norm,b_margins)



opt = torch.optim.AdamW(model.parameters(), lr=5e-5)



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
    
            loss = calculate_cosine_angle_loss(embeddings_1_data_norm_first, embeddings_1_diff_data_norm_first, embeddings_2_data_norm_first, embeddings_2_diff_data_norm_first,b_margins_first,int(EMBEDDING_SIZE/4))/BATCH_SIZE
            loss += calculate_cosine_angle_loss(embeddings_1_data_norm_second, embeddings_1_diff_data_norm_second, embeddings_2_data_norm_second, embeddings_2_diff_data_norm_second,b_margins_second,int(EMBEDDING_SIZE/2))/BATCH_SIZE
            loss += calculate_cosine_angle_loss(embeddings_1_data_norm_third, embeddings_1_diff_data_norm_third, embeddings_2_data_norm_third, embeddings_2_diff_data_norm_third,b_margins_third,int(EMBEDDING_SIZE/1))/BATCH_SIZE
            #loss += calculate_cosine_angle_loss(embeddings_1_data_norm_fourth, embeddings_1_diff_data_norm_fourth, embeddings_2_data_norm_fourth, embeddings_2_diff_data_norm_fourth,b_margins_fourth,int(EMBEDDING_SIZE))/BATCH_SIZE
            total_loss+=float(loss.item())
    
        return total_loss 




import os
if not os.path.isdir('multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular'):
    os.mkdir('multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular')
epoch = 0 



minimum_loss = 999999999999999
for epoch in range(0,1000000):
    num_batches = 0 
    model = model.train()
    for batch in tqdm(train_data_dataloader, desc=f'train-{epoch}'):
        b_ids_1, b_mask_1,b_ids_2, b_mask_2, b_margins = (batch['text_pair1_token_ids'],
                                   batch['text_pair1_attention_mask'], batch['text_pair2_token_ids'], batch['text_pair2_attention_mask'],batch['margins'])
        opt.zero_grad()
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


        

        loss = calculate_cosine_angle_loss(embeddings_1_data_norm_first, embeddings_1_diff_data_norm_first, embeddings_2_data_norm_first, embeddings_2_diff_data_norm_first,b_margins_first,int(EMBEDDING_SIZE/4))/BATCH_SIZE
        loss += calculate_cosine_angle_loss(embeddings_1_data_norm_second, embeddings_1_diff_data_norm_second, embeddings_2_data_norm_second, embeddings_2_diff_data_norm_second,b_margins_second,int(EMBEDDING_SIZE/2))/BATCH_SIZE
        loss += calculate_cosine_angle_loss(embeddings_1_data_norm_third, embeddings_1_diff_data_norm_third, embeddings_2_data_norm_third, embeddings_2_diff_data_norm_third,b_margins_third,int(EMBEDDING_SIZE/1))/BATCH_SIZE
        
        num_batches+=1
        

        loss.backward()
        opt.step()
        try:
            f = open('multilingual-matryoshka-e5-calculate_cosine_angle_loss-scne-loss-train_loss-fixed.txt','a+')
            f.write(str(loss.item())+"\n")
            f.close()
        except Exception as e:
            print(e)
    try:
        validation_loss = model_eval(val_data_dataloader,model, device) 
        if validation_loss < minimum_loss:
                try:
                    torch.save(model.state_dict(), 'multilingual-matryoshka-e5-calculate_cosine_angle_loss-fixed-regular/multilingual-20240816-matryoshka-e5-calculate_cosine_angle_loss-base'+str(epoch)+'-'+str(num_batches)+'.pt')
                except Exception as e:
                    print(e)
                minium_loss = validation_loss
    except Exception as e:
        print(e)





