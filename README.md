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

   
***Request mE5 Model Weights***

In this work, we find that a finetuned version of the mE5-base model achieved the best downstream results. To request the weights of the model used in this work, please fill out the following [Google form](https://forms.gle/ASzCcywsQ4Pd9Eyh6)

4. **Generate embeddings for raw text**

   ```bash
   python embedding_training_and_inference/inference.py \
     --input-path data/raw_texts.json \
     --output-path outputs/text_embeddings.json
   ```

5. **Cluster embeddings hierarchically**

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
```
@inproceedings{hanley-durumeric-2025-hierarchical,
    title = "Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings",
    author = "Hanley, Hans William Alexander  and
      Durumeric, Zakir",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.124/",
    doi = "10.18653/v1/2025.acl-long.124",
    pages = "2476--2492",
    ISBN = "979-8-89176-251-0",
    abstract = "Contextual large language model embeddings are increasingly utilized for topic modeling and clustering. However, current methods often scale poorly, rely on opaque similarity metrics, and struggle in multilingual settings. In this work, we present a novel, scalable, interpretable, hierarchical, and multilingual approach to clustering news articles and social media data. To do this, we first train multilingual Matryoshka embeddings that can determine story similarity at varying levels of granularity based on which subset of the dimensions of the embeddings is examined. This embedding model achieves state-of-the-art performance on the SemEval 2022 Task 8 test dataset (Pearson $\rho$ = 0.816). Once trained, we develop an efficient hierarchical clustering algorithm that leverages the hierarchical nature of Matryoshka embeddings to identify unique news stories, narratives, and themes. We conclude by illustrating how our approach can identify and cluster stories, narratives, and overarching themes within real-world news datasets."
}
```

## License and Copyright

Copyright 2024 The Board of Trustees of The Leland Stanford Junior University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
