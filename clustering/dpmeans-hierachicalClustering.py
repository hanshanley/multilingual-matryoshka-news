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

# Load embeddings and initialize necessary data structures
embeddings = []
old_urls = []

# Load embeddings from a pickle file
with open('/mnt/projects/qanon_proj/Spinda/testing-original-sts22-mat-aligned-multilingual-e5-base-20240814-embeddings.obj', 'rb') as f:
    ids_to_embeddings = pickle.load(f)

# Extract embeddings and their corresponding IDs
for id_ in ids_to_embeddings:
    embeddings.append(ids_to_embeddings[id_][0])
    old_urls.append(id_)

# Initialize mappings from IDs to indices and vice versa
ids = []
ids_to_ind = {}
ind_to_id = {}
ind = 0

# Populate the ID-to-index and index-to-ID mappings, along with the embeddings list
for id_ in ids_to_embeddings:
    ids.append(id_)
    ids_to_ind[id_] = ind
    ind_to_id[ind] = id_
    embeddings.append(ids_to_embeddings[id_][0])
    ind += 1

def add_index_to_hierachy(hierachy_index, new_hierachy_embedding, center_hierachy_to_index):
    """
    Add a new embedding to the hierarchical index.

    Parameters:
    - hierachy_index (int): The index in the hierarchy where the embedding will be added.
    - new_hierachy_embedding (np.array): The new embedding to be added.
    - center_hierachy_to_index (dict): Dictionary mapping hierarchy indices to FAISS indices.
    """
    d = len(new_hierachy_embedding)  # Dimensionality of the embedding
    quantizer = faiss.IndexFlatIP(d)  # Initialize the quantizer with inner product metric
    nlist = 1  # Number of clusters (for the index)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 1  # Number of clusters to search
    current_embeddings = np.array(new_hierachy_embedding).reshape(1, d).astype(np.float32)
    faiss.normalize_L2(current_embeddings)  # Normalize embeddings
    index.train(current_embeddings)  # Train the index
    index.add(current_embeddings)  # Add the embeddings to the index
    center_hierachy_to_index[hierachy_index] = index  # Update the mapping

def update_index_in_hierachy(hierachy_index, new_embedding_list, center_hierachy_to_index):
    """
    Update an existing hierarchical index with new embeddings.

    Parameters:
    - hierachy_index (int): The index in the hierarchy to be updated.
    - new_embedding_list (list): List of new embeddings to be added.
    - center_hierachy_to_index (dict): Dictionary mapping hierarchy indices to FAISS indices.
    """
    d = len(new_embedding_list[0])  # Dimensionality of the embeddings
    quantizer = faiss.IndexFlatIP(d)  # Initialize the quantizer with inner product metric
    nlist = max(int(np.sqrt(len(new_embedding_list)) / 10), 1)  # Number of clusters
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nlist  # Number of clusters to search
    current_embeddings = np.array(new_embedding_list).reshape(len(new_embedding_list), d).astype(np.float32)
    faiss.normalize_L2(current_embeddings)  # Normalize embeddings
    index.train(current_embeddings)  # Train the index
    index.add(current_embeddings)  # Add the embeddings to the index
    center_hierachy_to_index[hierachy_index] = index  # Update the mapping

def baseline_matched_indices(current_index, tested_embedding):
    """
    Perform a search in the index for the closest matching embeddings.

    Parameters:
    - current_index (FAISS index): The FAISS index to search.
    - tested_embedding (np.array): The embedding to search for.

    Returns:
    - current_matched_indices (list): List of matched indices.
    - indices_distances (list): Corresponding distances for the matched indices.
    """
    current_index = faiss.index_cpu_to_gpu(res, 0, current_index)  # Move index to GPU
    current_amount = 1024  # Number of top results to retrieve
    distance, indices = current_index.search(tested_embedding.reshape((len(tested_embedding), tested_embedding.shape[1])), current_amount)

    current_matched_indices = []
    indices_distances = []

    for ind2, dist in enumerate(distance):
        top_index_matched = indices[ind2][0]
        current_matched_indices.append(top_index_matched)
        indices_distances.append(dist[0])

    return current_matched_indices, indices_distances

def baseline_matched_indices_double_check(current_index, tested_embedding, threshold):
    """
    Perform a search in the index and double-check the top results against a threshold.

    Parameters:
    - current_index (FAISS index): The FAISS index to search.
    - tested_embedding (np.array): The embedding to search for.
    - threshold (float): The threshold to filter results.

    Returns:
    - current_matched_indices (list): List of matched indices, filtered by the threshold.
    - indices_distances (list): Corresponding distances for the matched indices.
    """
    current_index = faiss.index_cpu_to_gpu(res, 0, current_index)  # Move index to GPU
    current_amount = 2  # Retrieve the top 2 results
    distance, indices = current_index.search(tested_embedding.reshape((len(tested_embedding), tested_embedding.shape[1])), current_amount)

    current_matched_indices = []
    indices_distances = []

    for ind2, dist in enumerate(distance):
        if dist[1] > threshold:
            current_matched_indices.append(indices[ind2][1])
            indices_distances.append(dist[1])
        else:
            current_matched_indices.append(-1)  # No match found
            indices_distances.append(-1)  # No valid distance

    return current_matched_indices, indices_distances

def bottom_matched_indices(current_index, tested_embedding):
    """
    Perform a search in the index and retrieve the bottom matched indices.

    Parameters:
    - current_index (FAISS index): The FAISS index to search.
    - tested_embedding (np.array): The embedding to search for.

    Returns:
    - current_matched_indices (list): List of matched indices.
    - indices_distances (list): Corresponding distances for the matched indices.
    """
    current_index = faiss.index_cpu_to_gpu(res, 0, current_index)  # Move index to GPU
    current_amount = 1024  # Number of top results to retrieve
    distance, indices = current_index.search(tested_embedding.reshape((len(tested_embedding), tested_embedding.shape[1])), current_amount)

    current_matched_indices = []
    indices_distances = []

    for ind2, dist in enumerate(distance):
        top_index_matched = indices[ind2][0]
        current_matched_indices.append(top_index_matched)
        indices_distances.append(dist[0])

    return current_matched_indices, indices_distances

def baseline_matched_indices_all(current_index, tested_embedding):
    """
    Perform a search in the index and retrieve all matched indices above a threshold.

    Parameters:
    - current_index (FAISS index): The FAISS index to search.
    - tested_embedding (np.array): The embedding to search for.

    Returns:
    - current_matched_indices (list): List of matched indices that meet the threshold.
    - indices_distances (list): Corresponding distances for the matched indices.
    """
    current_index = faiss.index_cpu_to_gpu(res, 0, current_index)  # Move index to GPU
    current_amount = 1024  # Number of top results to retrieve
    distance, indices = current_index.search(tested_embedding.reshape((len(tested_embedding), tested_embedding.shape[1])), current_amount)

    current_matched_indices = []
    indices_distances = []

    for ind2, dist in enumerate(distance):
        test_indices = []
        test_distances = []
        ind3 = 0

        while dist[ind3] > THRESHOLD:
            test_indices.append(indices[ind2][ind3])
            test_distances.append(dist[ind3])
            ind3 += 1
            if ind3 == current_amount:
                break

        current_matched_indices.append(test_indices)
        indices_distances.append(test_distances)

    return current_matched_indices, indices_distances

def baseline_matched_indices_immediate(current_index, tested_embedding, threshold):
    """
    Perform a search in the index and immediately retrieve the matched indices above a threshold.

    Parameters:
    - current_index (FAISS index): The FAISS index to search.
    - tested_embedding (np.array): The embedding to search for.
    - threshold (float): The threshold to filter results.

    Returns:
    - current_matched_indices (list): List of matched indices that meet the threshold.
    - indices_distances (list): Corresponding distances for the matched indices.
    """
    current_index = faiss.index_cpu_to_gpu(res, 0, current_index)  # Move index to GPU
    current_amount = 1  # Retrieve the top result
    distance, indices = current_index.search(tested_embedding.reshape((len(tested_embedding), tested_embedding.shape[1])), current_amount)

    current_matched_indices = []
    indices_distances = []

    for ind2, dist in enumerate(distance):
        if dist[0] > threshold:
            current_matched_indices.append(indices[ind2][0])
            indices_distances.append(dist[0])
        else:
            current_matched_indices.append(-1)  # No match found
            indices_distances.append(-1)  # No valid distance

    return current_matched_indices, indices_distances

def compute_inertia(cosine_similarities):
    """
    Compute the inertia given a list or array of cosine similarities.

    Parameters:
    - cosine_similarities (list or array): A list or array of cosine similarities between
                                           data points and their corresponding cluster centroids.

    Returns:
    - float: The computed inertia.
    """
    cosine_distances = 1 - np.array(cosine_similarities)  # Convert cosine similarities to cosine distances
    squared_cosine_distances = cosine_distances ** 2  # Square the cosine distances
    inertia = np.sum(squared_cosine_distances)  # Sum of squared distances gives the inertia
    return inertia

def get_indices_above_threshold(values, threshold=0.6):
    """
    Retrieve indices of values above a certain threshold.

    Parameters:
    - values (list or array): The list or array of values to evaluate.
    - threshold (float): The threshold to filter values.

    Returns:
    - indices (array): Array of indices where the values exceed the threshold.
    """
    array = np.array(values)
    indices = np.where(array > threshold)[0]
    return indices

def create_index(embeddings):
    """
    Create a FAISS index from a list of embeddings.

    Parameters:
    - embeddings (list): List of embeddings to be indexed.

    Returns:
    - index (FAISS index): The created FAISS index.
    """
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    gc.collect()  # Clean up garbage collection
    d = len(embeddings[0])  # Dimensionality of the embeddings
    quantizer = faiss.IndexFlatIP(d)  # Initialize the quantizer with inner product metric
    nlist = max(int(np.sqrt(len(embeddings))), 1)  # Number of clusters for the index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nlist  # Number of clusters to search
    current_embeddings = np.array(embeddings).astype(np.float32)
    faiss.normalize_L2(current_embeddings)  # Normalize embeddings
    index.train(current_embeddings)  # Train the index
    index.add(current_embeddings)  # Add the embeddings to the index
    return index
