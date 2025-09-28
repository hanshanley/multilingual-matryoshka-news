'''
Hierarchical Clustering of Embeddings Using RAC++ with Multi-Layer Centroid Calculation and Matryoshka Embeddings

This script performs hierarchical clustering on high-dimensional embeddings using the RAC++ clustering algorithm.
The process involves three layers of clustering, where embeddings are first grouped and then progressively
refined through the computation of centroids at each layer. 

See Methodology Section for Additional details. 
'''
# Import necessary libraries
import racplusplus
import pickle
import numpy as np
import faiss

# Set the dimension of embeddings
dim = 768

# Load embeddings from a file
with open('EMBDDING FILE', 'rb') as f:
    complete_ids_to_embeddings = pickle.load(f)

# Initialize dictionaries and lists to store indices and embeddings
ids_to_inds = {}
embeddings = []
ind = 0

# Map each ID to its corresponding embeddings and store them with indices
for id_ in complete_ids_to_embeddings:
    if id_ not in ids_to_inds:
        ids_to_inds[id_] = []
    for embedding in complete_ids_to_embeddings[id_]:
        ids_to_inds[id_].append(ind)
        embeddings.append(embedding)
        ind += 1

# Convert embeddings list to a NumPy array and normalize
embeddings = np.array(embeddings).reshape(len(embeddings), dim)
top_embeddings = np.array(embeddings)[,:192].reshape(len(embeddings), 192)
faiss.normalize_L2(embeddings)

# Set the thresholds for the three clustering layers
THRESHOLDS = [0.5, 0.5, 0.5]

# Perform the first round of clustering using RAC++
labels = racplusplus.rac(top_embeddings, 1 - THRESHOLDS[0], None, 1000, 8, "cosine")

# Initialize structures to map clusters to labels
new_clusters = []
clusters_to_labels = {}
ind = 0

# Group embeddings by their assigned cluster labels
for label in labels:
    if label not in clusters_to_labels:
        clusters_to_labels[label] = []
    clusters_to_labels[label].append(ind)
    ind += 1

# Compute the average embedding for each cluster (first layer)
original_labels = []
for cluster in clusters_to_labels:
    new_clusters.append(np.array(np.average(embeddings[clusters_to_labels[cluster]], axis=0)[:384]))
    original_labels.append(cluster)

# Perform the second round of clustering on the centroids of the first layer clusters
next_labels = racplusplus.rac(new_clusters, 1 - THRESHOLDS[1], None, 1000, 8, "cosine")

# Map first layer clusters to second layer clusters
first_layer_to_second_layer = {}
first_cluster = 0
for next_label in next_labels:
    first_layer_to_second_layer[original_labels[first_cluster]] = next_label
    first_cluster += 1

# Map original indices to their corresponding second layer clusters
original_ind_to_second_cluster = {}
for first_layer in first_layer_to_second_layer:
    second_layer_cluster = first_layer_to_second_layer[first_layer]
    for original_ind in clusters_to_labels[first_layer]:
        original_ind_to_second_cluster[original_ind] = second_layer_cluster

# Initialize structures to map second layer clusters to labels
new_clusters = []
second_clusters_to_labels = {}
ind = 0

# Group original indices by their assigned second layer cluster labels
for ind in original_ind_to_second_cluster:
    label = original_ind_to_second_cluster[ind]
    if label not in second_clusters_to_labels:
        second_clusters_to_labels[label] = []
    second_clusters_to_labels[label].append(ind)

# Compute the average embedding for each cluster (second layer)
second_labels = []
for cluster in second_clusters_to_labels:
    new_clusters.append(np.array(np.average(embeddings[second_clusters_to_labels[cluster]], axis=0)[:768]))
    second_labels.append(cluster)

# Perform the third round of clustering on the centroids of the second layer clusters
final_labels = racplusplus.rac(new_clusters, 1 - THRESHOLDS[2], None, 1000, 8, "cosine")

# Map second layer clusters to third layer clusters
second_layer_to_third_layer = {}
second_cluster = 0
for final_label in final_labels:
    second_layer_to_third_layer[second_labels[second_cluster]] = final_label
    second_cluster += 1

# Map original indices to their corresponding third layer clusters
original_ind_to_third_cluster = {}
for first_layer in first_layer_to_second_layer:
    second_layer_cluster = first_layer_to_second_layer[first_layer]
    third_layer_cluster = second_layer_to_third_layer[second_layer_cluster]
    for original_ind in clusters_to_labels[first_layer]:
        original_ind_to_third_cluster[original_ind] = third_layer_cluster
