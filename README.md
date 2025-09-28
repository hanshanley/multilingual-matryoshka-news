# Hierarchical Level-Wise News Clustering via Multilingual Matryoshka Embeddings

This repository packages three key components for building and evaluating
Matryoshka-style news clustering systems:

1. **Embedding training** – Matryoshka and AngIE objectives for generating
   multi-resolution sentence embeddings.
2. **Inference utilities** – Batch scripts for embedding arbitrary text using
   the trained encoders (or any Hugging Face checkpoint).
3. **Hierarchical clustering** – A RAC++-based pipeline that clusters
   embeddings at multiple abstraction levels (e.g., 192 → 384 → 768
   dimensions) to recover news story hierarchies.

The workflow mirrors the approach described in the paper, “Hierarchical
Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings”.

## Repository Layout

```
embedding_training_and_inference/
  ├── matryoshka-angie.py      # Train Matryoshka embeddings with multi-level losses
  ├── modified-angie.py        # Train AngIE-style baseline embeddings
  ├── inference.py             # Generate embeddings for arbitrary ID/text pairs
  └── README.md                # Detailed usage documentation for the scripts above

clustering/
  ├── level-wise-rac.py        # RAC++ hierarchical clustering CLI (192/384/768 schedule)
  └── README.md                # Instructions and requirements for clustering

data/
  └── readme.md                # Notes on obtaining datasets (SemEval 2022, Miranda et al., 20 Newsgroups)

requirements.txt               # Core Python dependencies
utils.py                       # Helper functions shared across the project
```

## Quick Start

1. **Install dependencies**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   # Install FAISS and RAC++ separately if you plan to run clustering
   ```

2. **Train embeddings**

   ```bash
   python embedding_training_and_inference/matryoshka-angie.py \
     --train-path data/train_pairs.jsonl \
     --val-path data/dev_pairs.jsonl \
     --output-dir outputs/matryoshka \
     --labels-path data/labels_semeval_2022_task_eight.json
   ```

   For a non-Matryoshka baseline, switch to `modified-angie.py`.

3. **Generate embeddings for raw text**

   ```bash
   python embedding_training_and_inference/inference.py \
     --input-path data/raw_texts.json \
     --output-path outputs/text_embeddings.json
   ```

4. **Cluster embeddings hierarchically**

   ```bash
   python clustering/level-wise-rac.py \
     --embedding-path outputs/text_embeddings.pkl \
     --output-path outputs/cluster_assignments.json
   ```

   Thresholds and dimensions default to the Matryoshka schedule but can be
   overridden via CLI flags.

Refer to the README files in each folder for additional options, required
data formats, and detailed explanations of the batching/sampling strategies.

## Citation

If you use this repository in academic work, please cite the original paper.
