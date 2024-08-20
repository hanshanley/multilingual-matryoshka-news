
#!/usr/bin/env python
# coding: utf-8

import os
import faiss
import json
import random
import gc
import math
import numpy as np
import pickle

# Load embeddings from a file and prepare necessary structures
embeddings = []
old_urls = []

# Load the precomputed embeddings from a file
with open('/mnt/projects/qanon_proj/Spinda/testing-original-sts22-mat-aligned-multilingual-e5-base-20240814-embeddings.obj', 'rb') as f:
    ids_to_embeddings = pickle.load(f)

# Extract embeddings and corresponding URLs (or IDs)
for id_ in ids_to_embeddings:
    embeddings.append(ids_to_embeddings[id_][0])
    old_urls.append(id_)

# Initialize mappings for IDs to indices and vice versa
ids = []
ids_to_ind = {}
ind_to_id = {}
embeddings = []
ind = 0

# Populate the ID-to-index and index-to-ID mappings, as well as the embeddings list
for id_ in ids_to_embeddings:
    ids.append(id_)
    ids_to_ind[id_] = ind
    ind_to_id[ind] = id_
    embeddings.append(ids_to_embeddings[id_][0])
    ind += 1

def add_index_to_hierachy(hierachy_index, new_hierachy_embedding, center_hierachy_to_index):
    """
    Adds a new embedding to the hierarchical index.
    
    Parameters:
        hierachy_index (int): The index in the hierarchy where the embedding is added.
        new_hierachy_embedding (np.array): The new embedding to add.
        center_hierachy_to_index (dict): The dictionary mapping hierarchy indices to FAISS indices.
    """
    d = len(new_hierachy_embedding)  # Dimensionality of the embedding
    quantizer = faiss.IndexFlatIP(d)  # Quantizer for the FAISS index
    nlist = 1  # Number of clusters in the index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 1  # Number of clusters to search
    current_embeddings = np.array(new_hierachy_embedding).reshape(1, d).astype(np.float32)
    faiss.normalize_L2(current_embeddings)  # Normalize the embeddings
    index.train(current_embeddings)  # Train the index with the embedding
    index.add(current_embeddings)  # Add the embedding to the index
    center_hierachy_to_index[hierachy_index] = index  # Update the hierarchy index

def update_index_in_hierachy(hierachy_index, new_embedding_list, center_hierachy_to_index):
    """
    Updates the existing hierarchical index with new embeddings.
    
    Parameters:
        hierachy_index (int): The index in the hierarchy to update.
        new_embedding_list (list): A list of new embeddings to add to the index.
        center_hierachy_to_index (dict): The dictionary mapping hierarchy indices to FAISS indices.
    """
    d = len(new_embedding_list[0])  # Dimensionality of the embeddings
    quantizer = faiss.IndexFlatIP(d)  # Quantizer for the FAISS index
    nlist = max(int(np.sqrt(len(new_embedding_list)) / 10), 1)  # Number of clusters in the index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nlist  # Number of clusters to search
    current_embeddings = np.array(new_embedding_list).reshape(len(new_embedding_list), d).astype(np.float32)
    faiss.normalize_L2(current_embeddings)  # Normalize the embeddings
    index.train(current_embeddings)  # Train the index with the new embeddings
    index.add(current_embeddings)  # Add the new embeddings to the index
    center_hierachy_to_index[hierachy_index] = index  # Update the hierarchy index
