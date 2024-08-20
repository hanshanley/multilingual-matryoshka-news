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
with open('LOCATION OF EMBEDDING', 'rb') as f:
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




# Initialize dictionaries to store hierarchical structures
center_hierachy = {}
center_hierachy_to_embeddings = {}
center_hierachy_to_index = {}

# Create the first level of hierarchy with embeddings of size 192
center_hierachy_to_embeddings['level1'] = [embeddings[0][:192].reshape(1, 192)]
faiss.normalize_L2(center_hierachy_to_embeddings['level1'][0])
center_hierachy_to_embeddings['level1'][0] = center_hierachy_to_embeddings['level1'][0].reshape(192,)

# Create the second level of hierarchy with embeddings of size 384
center_hierachy_to_embeddings['level2'] = [embeddings[0][:384].reshape(1, 384)]
faiss.normalize_L2(center_hierachy_to_embeddings['level2'][0])
center_hierachy_to_embeddings['level2'][0] = center_hierachy_to_embeddings['level2'][0].reshape(384,)

# Generate a random set of indices to start with
current_indices = set(range(len(embeddings)))
old_random_indices = np.random.choice(list(current_indices), min(30, len(current_indices)), replace=False)

# Reshape embeddings for the third level hierarchy (full embedding size)
old_random_indices = np.array(old_random_indices)
embeddings = np.array(embeddings)
center_hierachy_to_embeddings['level3'] = embeddings[old_random_indices].reshape(len(old_random_indices), 768)
faiss.normalize_L2(center_hierachy_to_embeddings['level3'])
center_hierachy_to_embeddings['level3'] = [center_hierachy_to_embeddings['level3'][i] for i in range(center_hierachy_to_embeddings['level3'].shape[0])]

# Initialize GPU resources for FAISS
res = faiss.StandardGpuResources()
gc.collect()

# Create FAISS indices for each level of the hierarchy
for hier in center_hierachy_to_embeddings:
    d = len(center_hierachy_to_embeddings[hier][0])
    quantizer = faiss.IndexFlatIP(d)
    nlist = 1  # Number of clusters for the index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 1  # Number of clusters to search
    current_embeddings = np.array(center_hierachy_to_embeddings[hier]).reshape(len(center_hierachy_to_embeddings[hier]), d)
    current_embeddings = np.array(current_embeddings, dtype=np.float32)
    faiss.normalize_L2(current_embeddings)
    index.train(current_embeddings)
    index.add(current_embeddings)
    center_hierachy_to_index[hier] = index

# Initialize dictionaries for layer indices
indices_to_first_layer = {}
indices_to_second_layer = {}
indices_to_third_layer = {}

# Deep copy embeddings for processing
deep_copied_embeddings = np.array(embeddings).copy()

# Initialize variables for clustering process
current_baseline_index_to_check = 0
similarities = [0.5,0.5,0.5]  # Similarity thresholds for the layers

# Initialize variables for the iterative clustering process
domain_similarity_url_url_sim = {}
index_to_assignment = {}
num_done = 0 
ind = 0 
current_embeddings = np.array(embeddings).reshape(len(embeddings), 768)
all_indices = set(range(len(current_embeddings)))
current_indices = set(range(len(current_embeddings)))
already_assigned = set()

# Print the total number of embeddings
print(len(current_embeddings))

# Initialize the completed flag
completed = False

# Number of random indices to select
num_indices = 32768

# Initialize lists and variables for the inertia calculation
inertias = []
alpha = 0.1
ewa_inerta = 0  # Exponential weighted average of inertia
num_not_improved = 0 
min_ewa = 0  # Minimum exponential weighted average inertia
current_baseline_index_to_check = 0  # Index of the current baseline to check
update_flags = [False] * 4  # Flags for updating different levels
previous_distances = []

# Select a random set of unique indices for testing
old_random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)

# Prepare test embeddings for the third level hierarchy
test_tested_embeddings_level3 = np.ascontiguousarray(np.array(current_embeddings[old_random_indices])[:, :768])
faiss.normalize_L2(test_tested_embeddings_level3)

# Initialize flags for control flow in the clustering loop
last_check = False
should_do = False

# Main loop for hierarchical clustering
while len(current_indices) > 0:
    # Get the current similarity threshold based on the baseline index
    THRESHOLD = similarities[current_baseline_index_to_check]

    # Select a random set of indices for the current iteration
    random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)

    # Prepare test embeddings for all three levels of the hierarchy
    total_tested_embeddings_level1 = np.ascontiguousarray(np.array(current_embeddings[random_indices])[:, :192])
    total_tested_embeddings_level2 = np.ascontiguousarray(np.array(current_embeddings[random_indices])[:, :384])
    total_tested_embeddings_level3 = np.ascontiguousarray(np.array(current_embeddings[random_indices])[:, :768])
    faiss.normalize_L2(total_tested_embeddings_level1)
    faiss.normalize_L2(total_tested_embeddings_level2)
    faiss.normalize_L2(total_tested_embeddings_level3)

    # Perform a search on the third level of the hierarchy
    current_matched_indices, baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level3"], total_tested_embeddings_level3)
    _, test_baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level3"], total_tested_embeddings_level3)

    final_indices = []

    # Check if further refinement is needed
    if len(random_indices) < num_indices and not last_check:
        lower_indices = np.where(np.array(baseline_indices_distances) < similarities[current_baseline_index_to_check])[0]
        self_index = create_index(total_tested_embeddings_level3)
        self_current_matched_indices, sims = baseline_matched_indices_double_check(self_index, total_tested_embeddings_level3[lower_indices], similarities[current_baseline_index_to_check])
        self_indices = np.where(np.array(self_current_matched_indices) == -1)[0]
        print(self_indices)
        if len(self_indices) > 0:
            final_indices = lower_indices[list(self_indices)]
        else:
            final_indices = []
        last_check = True
        should_do = True

    # Sort indices based on distance and compute inertia
    sorted_indices = np.argsort(baseline_indices_distances)
    print("MINIMUM SIM: " + str(baseline_indices_distances[sorted_indices[0]]))
    inertia = compute_inertia(test_baseline_indices_distances)
    inertia += np.power(1 - THRESHOLD, 2) * len(test_baseline_indices_distances)
    print("CURRENT LEVEL: " + str(current_baseline_index_to_check))
    print("COST: " + str(inertia))
    print("AVERAGE SIM: " + str(np.average(baseline_indices_distances)))
    print("num_not_improved: " + str(num_not_improved))

    # Update exponential weighted average inertia
    if len(inertias) == 0:
        ewa_inerta = inertia
        min_ewa = ewa_inerta
        print("MIN EWA: " + str(min_ewa))
    else:
        ewa_inerta = ewa_inerta * (1 - alpha) + inertia * alpha

    # Check if the current inertia is improving
    if ewa_inerta > min_ewa:
        print("CURRENT MIN EWA: " + str(min_ewa))
        print("CURRENT EWA: " + str(ewa_inerta))
        num_not_improved += 1
    else:
        min_ewa = ewa_inerta
        num_not_improved = 0

    inertias.append(inertia)
    print(ewa_inerta)

    # Update hierarchy if similarity threshold is met
    for current_index in sorted_indices[:1]:
        if baseline_indices_distances[current_index] < similarities[current_baseline_index_to_check]:
            current_indicies_to_do = set([current_index])
            if should_do:
                current_indicies_to_do.update(list(final_indices))
            should_do = False
            for more_index in current_indicies_to_do:
                tested_embeddings_level3 = total_tested_embeddings_level3[more_index].reshape(1, total_tested_embeddings_level3.shape[1])
                center_hierachy_to_embeddings['level3'].append(tested_embeddings_level3[0])
            update_index_in_hierachy("level3", center_hierachy_to_embeddings['level3'], center_hierachy_to_index)

    # Remove processed indices from the current set
    removed_indexes = set(get_indices_above_threshold(baseline_indices_distances, similarities[current_baseline_index_to_check]))
    removed_indexes.add(sorted_indices[0])
    if len(final_indices) > 0:
        removed_indexes.update(final_indices)
    current_indices.difference_update(random_indices[list(removed_indexes)])
    print(len(current_indices))
    print("NUM ABOVE: " + str(len(removed_indexes)))
    print("INDICES TO GO: " + str(len(current_indices)))

    if len(removed_indexes) == 1:
        last_check = False

# Build the first layer assignments based on level 3 clusters
indices_to_first_layer = {}
current_embeddings = np.array(embeddings).reshape(len(embeddings), 768)
all_indices = list(range(len(current_embeddings)))
BATCH_SIZE = 32768
new_ind = 0
first_layer_to_indices = {}

# Process embeddings in batches to create the first layer of assignments
for indices in range(len(current_embeddings) // BATCH_SIZE + 1):
    total_tested_embeddings_level3 = np.ascontiguousarray(np.array(current_embeddings[BATCH_SIZE * new_ind:BATCH_SIZE * (new_ind + 1)])[:, :768])
    faiss.normalize_L2(total_tested_embeddings_level3)
    baseline_indices, baseline_indices_distances = baseline_matched_indices_immediate(center_hierachy_to_index["level3"], total_tested_embeddings_level3, 0.4)
    new_index = 0
    for new_index in range(len(baseline_indices_distances)):
        indices_to_first_layer[BATCH_SIZE * new_ind + new_index] = baseline_indices[new_index][0]
        if baseline_indices[new_index][0] not in first_layer_to_indices:
            first_layer_to_indices[baseline_indices[new_index][0]] = []
        first_layer_to_indices[baseline_indices[new_index][0]].append(BATCH_SIZE * new_ind + new_index)
        new_index += 1
    new_ind += 1
    print(len(indices_to_first_layer))

# Compute centroids for the second layer
level2_centroids = {}
layer2_embeddings = []
for layer in first_layer_to_indices:
    level2_centroids[layer] = np.array(np.average(embeddings[first_layer_to_indices[layer]], axis=0)[:384]).reshape(1, 384)
    faiss.normalize_L2(level2_centroids[layer])
    layer2_embeddings.append(level2_centroids[layer][0])
    level2_centroids[layer] = level2_centroids[layer][0]

# Update the second level hierarchy with new centroids
center_hierachy_to_embeddings['level2'] = [level2_centroids[layer] for layer in first_layer_to_indices]
update_index_in_hierachy("level2", center_hierachy_to_embeddings['level2'], center_hierachy_to_index)

# Initialize the second layer clustering process
num_indices = 32768
inertias = []
current_embeddings = np.array(layer2_embeddings).reshape(len(layer2_embeddings), 384)
current_indices = set(range(len(current_embeddings)))
current_baseline_index_to_check = 1

# Random selection of embeddings for testing at level 2
old_random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)
test_tested_embeddings_level2 = np.ascontiguousarray(np.array(current_embeddings[old_random_indices])[:, :384])
faiss.normalize_L2(test_tested_embeddings_level2)
last_check = False
should_do = False

# Main loop for the second layer clustering
while len(current_indices) > 0:
    THRESHOLD = similarities[current_baseline_index_to_check]
    random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)
    total_tested_embeddings_level2 = np.ascontiguousarray(np.array(current_embeddings[random_indices])[:, :384])
    faiss.normalize_L2(total_tested_embeddings_level2)
    current_matched_indices, baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level2"], total_tested_embeddings_level2)
    _, test_baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level2"], total_tested_embeddings_level2)

    final_indices = []
    if len(random_indices) < num_indices and not last_check:
        lower_indices = np.where(np.array(baseline_indices_distances) < similarities[current_baseline_index_to_check])[0]
        self_index = create_index(total_tested_embeddings_level2)
        self_current_matched_indices, sims = baseline_matched_indices_double_check(self_index, total_tested_embeddings_level2[lower_indices], similarities[current_baseline_index_to_check])
        self_indices = np.where(np.array(self_current_matched_indices) == -1)[0]
        print(self_indices)
        if len(self_indices) > 0:
            final_indices = lower_indices[list(self_indices)]
        else:
            final_indices = []
        last_check = True
        should_do = True

    sorted_indices = np.argsort(baseline_indices_distances)
    print("MINIMUM SIM: " + str(baseline_indices_distances[sorted_indices[0]]))
    inertia = compute_inertia(test_baseline_indices_distances)
    inertia += np.power(1 - THRESHOLD, 2) * len(test_baseline_indices_distances)
    print("CURRENT LEVEL: " + str(current_baseline_index_to_check))
    print("COST: " + str(inertia))
    print("AVERAGE SIM: " + str(np.average(baseline_indices_distances)))
    print("num_not_improved: " + str(num_not_improved))

    if len(inertias) == 0:
        ewa_inerta = inertia
        min_ewa = ewa_inerta
        print("MIN EWA: " + str(min_ewa))
    else:
        ewa_inerta = ewa_inerta * (1 - alpha) + inertia * alpha

    if ewa_inerta > min_ewa:
        print("CURRENT MIN EWA: " + str(min_ewa))
        print("CURRENT EWA: " + str(ewa_inerta))
        num_not_improved += 1
        if num_not_improved == 15:
            print('NOT IMPROVING')
            current_baseline_index_to_check += 1
            ewa_inerta = 0
            num_not_improved = 0
            min_ewa = 0
            inertias = []
            all_indices = set(range(len(current_embeddings)))
            current_indices = set(range(len(current_embeddings)))
            if current_baseline_index_to_check == 4:
                break
            continue
    else:
        min_ewa = ewa_inerta
        num_not_improved = 0

    inertias.append(inertia)
    print(ewa_inerta)

    for current_index in sorted_indices[:1]:
        if baseline_indices_distances[current_index] < similarities[current_baseline_index_to_check]:
            current_indicies_to_do = set([current_index])
            if should_do:
                current_indicies_to_do.update(list(final_indices))
            should_do = False
            for more_index in current_indicies_to_do:
                tested_embeddings_level2 = total_tested_embeddings_level2[more_index].reshape(1, total_tested_embeddings_level2.shape[1])
                center_hierachy_to_embeddings['level2'].append(tested_embeddings_level2[0])
            update_index_in_hierachy("level2", center_hierachy_to_embeddings['level2'], center_hierachy_to_index)

    removed_indexes = set(get_indices_above_threshold(baseline_indices_distances, similarities[current_baseline_index_to_check]))
    removed_indexes.add(sorted_indices[0])
    if len(final_indices) > 0:
        removed_indexes.update(final_indices)
    current_indices.difference_update(random_indices[list(removed_indexes)])
    print(len(current_indices))
    print("NUM ABOVE: " + str(len(removed_indexes)))
    print("INDICES TO GO: " + str(len(current_indices)))

    if len(removed_indexes) == 1:
        last_check = False

# Assign indices to the second layer based on level 2 clusters
indices_to_second_layer = {}
current_embeddings = np.array(layer2_embeddings).reshape(len(layer2_embeddings), 384)
all_indices = list(range(len(current_embeddings)))
BATCH_SIZE = 32768
new_ind = 0
second_layer_to_indices = {}

# Process embeddings in batches to create the second layer of assignments
for indices in range(len(current_embeddings) // BATCH_SIZE + 1):
    total_tested_embeddings_level2 = np.ascontiguousarray(np.array(current_embeddings[BATCH_SIZE * new_ind:BATCH_SIZE * (new_ind + 1)])[:, :384])
    faiss.normalize_L2(total_tested_embeddings_level2)
    baseline_indices, baseline_indices_distances = baseline_matched_indices_immediate(center_hierachy_to_index["level2"], total_tested_embeddings_level2, 0.0)
    new_index = 0
    for new_index in range(len(baseline_indices_distances)):
        if len(baseline_indices[new_index]) > 0:
            indices_to_second_layer[BATCH_SIZE * new_ind + new_index] = baseline_indices[new_index][0]
            if baseline_indices[new_index][0] not in second_layer_to_indices:
                second_layer_to_indices[baseline_indices[new_index][0]] = []
            second_layer_to_indices[baseline_indices[new_index][0]].append(BATCH_SIZE * new_ind + new_index)
        else:
            print('HERE')
            print(BATCH_SIZE * new_ind + new_index)
            indices_to_second_layer[BATCH_SIZE * new_ind + new_index] = []
    new_ind += 1
    print(len(indices_to_second_layer))

# Compute centroids for the first layer
level1_centroids = {}
layer1_embeddings = []
for layer2 in second_layer_to_indices:
    current_embeddings = []
    for layer1 in second_layer_to_indices[layer2]:
        current_embeddings.extend(embeddings[first_layer_to_indices[layer1]])
    num = len(current_embeddings)
    current_embeddings = np.array(current_embeddings).reshape(num, 768)
    level1_centroids[layer2] = np.average(current_embeddings[:, :192], axis=0).reshape(1, 192)
    faiss.normalize_L2(level1_centroids[layer2])
    level1_centroids[layer2] = level1_centroids[layer2][0]
    layer1_embeddings.append(level1_centroids[layer2])

# Update the first level hierarchy with new centroids
center_hierachy_to_embeddings['level1'] = [level1_centroids[layer2] for layer2 in second_layer_to_indices]
update_index_in_hierachy("level1", center_hierachy_to_embeddings['level1'], center_hierachy_to_index)

# Initialize the first layer clustering process
num_indices = 32768
inertias = []
current_embeddings = np.array(layer1_embeddings).reshape(len(layer1_embeddings), 192)
current_indices = set(range(len(current_embeddings)))
current_baseline_index_to_check = 2

# Random selection of embeddings for testing at level 1
old_random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)
test_tested_embeddings_level1 = np.ascontiguousarray(np.array(current_embeddings[old_random_indices])[:, :192])
faiss.normalize_L2(test_tested_embeddings_level1)
last_check = False
should_do = False

# Main loop for the first layer clustering
while len(current_indices) > 0:
    THRESHOLD = similarities[current_baseline_index_to_check]
    random_indices = np.random.choice(list(current_indices), min(num_indices, len(current_indices)), replace=False)
    total_tested_embeddings_level1 = np.ascontiguousarray(np.array(current_embeddings[random_indices])[:, :192])
    faiss.normalize_L2(total_tested_embeddings_level1)
    current_matched_indices, baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level1"], total_tested_embeddings_level1)
    _, test_baseline_indices_distances = baseline_matched_indices(center_hierachy_to_index["level1"], total_tested_embeddings_level1)

    final_indices = []
    if len(random_indices) < num_indices and not last_check:
        lower_indices = np.where(np.array(baseline_indices_distances) < similarities[current_baseline_index_to_check])[0]
        self_index = create_index(total_tested_embeddings_level1)
        self_current_matched_indices, sims = baseline_matched_indices_double_check(self_index, total_tested_embeddings_level1[lower_indices], similarities[current_baseline_index_to_check])
        self_indices = np.where(np.array(self_current_matched_indices) == -1)[0]
        print(self_indices)
        if len(self_indices) > 0:
            final_indices = lower_indices[list(self_indices)]
        else:
            final_indices = []
        last_check = True
        should_do = True

    sorted_indices = np.argsort(baseline_indices_distances)
    print("MINIMUM SIM: " + str(baseline_indices_distances[sorted_indices[0]]))
    inertia = compute_inertia(test_baseline_indices_distances)
    inertia += np.power(1 - THRESHOLD, 2) * len(test_baseline_indices_distances)
    print("CURRENT LEVEL: " + str(current_baseline_index_to_check))
    print("COST: " + str(inertia))
    print("AVERAGE SIM: " + str(np.average(baseline_indices_distances)))
    print("num_not_improved: " + str(num_not_improved))

    if len(inertias) == 0:
        ewa_inerta = inertia
        min_ewa = ewa_inerta
        print("MIN EWA: " + str(min_ewa))
    else:
        ewa_inerta = ewa_inerta * (1 - alpha) + inertia * alpha

    if ewa_inerta > min_ewa:
        print("CURRENT MIN EWA: " + str(min_ewa))
        print("CURRENT EWA: " + str(ewa_inerta))
        num_not_improved += 1
        if num_not_improved == 15:
            print('NOT IMPROVING')
            current_baseline_index_to_check += 1
            ewa_inerta = 0
            num_not_improved = 0
            min_ewa = 0
            inertias = []
            all_indices = set(range(len(current_embeddings)))
            current_indices = set(range(len(current_embeddings)))
            if current_baseline_index_to_check == 4:
                break
            continue
    else:
        min_ewa = ewa_inerta
        num_not_improved = 0

    inertias.append(inertia)
    print(ewa_inerta)

    for current_index in sorted_indices[:1]:
        if baseline_indices_distances[current_index] < similarities[current_baseline_index_to_check]:
            current_indicies_to_do = set([current_index])
            if should_do:
                current_indicies_to_do.update(list(final_indices))
            should_do = False
            for more_index in current_indicies_to_do:
                tested_embeddings_level1 = total_tested_embeddings_level1[more_index].reshape(1, total_tested_embeddings_level1.shape[1])
                center_hierachy_to_embeddings['level1'].append(tested_embeddings_level1[0])
            if len(center_hierachy_to_embeddings['level1']) == 140:
                print(total_tested_embeddings_level1)
            update_index_in_hierachy("level1", center_hierachy_to_embeddings['level1'], center_hierachy_to_index)

    removed_indexes = set(get_indices_above_threshold(baseline_indices_distances, similarities[current_baseline_index_to_check]))
    removed_indexes.add(sorted_indices[0])
    if len(final_indices) > 0:
        removed_indexes.update(final_indices)
    current_indices.difference_update(random_indices[list(removed_indexes)])
    print(len(current_indices))
    print("NUM ABOVE: " + str(len(removed_indexes)))
    print("INDICES TO GO: " + str(len(current_indices)))

    if len(removed_indexes) == 1:
        last_check = False

# Assign indices to the last layer based on level 1 clusters
indices_to_last_layer = {}
current_embeddings = np.array(layer1_embeddings).reshape(len(layer1_embeddings), 192)
all_indices = list(range(len(current_embeddings)))
BATCH_SIZE = 32768
new_ind = 0
last_layer_to_indices = {}

# Process embeddings in batches to create the last layer of assignments
for indices in range(len(current_embeddings) // BATCH_SIZE + 1):
    total_tested_embeddings_level1 = np.ascontiguousarray(np.array(current_embeddings[BATCH_SIZE * new_ind:BATCH_SIZE * (new_ind + 1)])[:, :192])
    faiss.normalize_L2(total_tested_embeddings_level1)
    baseline_indices, baseline_indices_distances = baseline_matched_indices_immediate(center_hierachy_to_index["level1"], total_tested_embeddings_level1, 0.2)
    new_index = 0
    for new_index in range(len(baseline_indices_distances)):
        if len(baseline_indices[new_index]) > 0:
            indices_to_last_layer[BATCH_SIZE * new_ind + new_index] = baseline_indices[new_index][0]
            if baseline_indices[new_index][0] not in last_layer_to_indices:
                last_layer_to_indices[baseline_indices[new_index][0]] = []
            last_layer_to_indices[baseline_indices[new_index][0]].append(BATCH_SIZE * new_ind + new_index)
        else:
            print('HERE')
            last_layer_to_indices[baseline_indices[new_index][0]] = []
    new_ind += 1
    print(len(last_layer_to_indices))

# Compute centroids for the final (level 0) layer
level0_centroids = {}
layer0_embeddings = []
for layer3 in last_layer_to_indices:
    current_embeddings = []
    for layer2 in last_layer_to_indices[layer3]:
        if layer2 in second_layer_to_indices:
            for layer1 in second_layer_to_indices[layer2]:
                current_embeddings.extend(embeddings[first_layer_to_indices[layer1]])
    num = len(current_embeddings)
    if num > 0:
        current_embeddings = np.array(current_embeddings).reshape(num, 768)
        level0_centroids[layer3] = np.average(current_embeddings[:, :192], axis=0).reshape(1, 192)
        faiss.normalize_L2(level0_centroids[layer3])
        level0_centroids[layer3] = level0_centroids[layer3][0]
        layer0_embeddings.append(level0_centroids[layer3])
    else:
        print('HERE')

