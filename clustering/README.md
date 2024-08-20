# Code for Clustering Matryoksha Embeddings 

To utilize this code, download the faiss-gpu (version=1.8.0) and the racplusplus library https://github.com/porterehunley/RACplusplus


This project implements hierarchical clustering on a large set of embeddings using FAISS (Facebook AI Similarity Search) and the RACPLUSPLUS for efficient similarity search and clustering. The goal is to organize embeddings into multiple hierarchical layers, each representing a different level of abstraction in the clustering process using Matryoshka embeddings as a proxy for the different level of abstraction. The DP-Means script optimized for performance using GPU acceleration.
