# Hierarchical Clustering Utilities

This folder contains a CLI for recursively clustering Matryoshka-style embeddings with [RAC++](https://github.com/porterehunley/RACplusplus) and FAISS normalisation. The pipeline mirrors the nesting used by Matryoshka encoders: it clusters the full embeddings, builds centroids for each cluster, optionally truncates their dimensionality, and repeats the process for deeper abstraction levels.

## Requirements

- Python 3.8+
- `racplusplus` (install from the repository linked above)
- `faiss-cpu` **or** `faiss-gpu`
- `numpy`

You can install the Python dependencies listed in the project-wide `requirements.txt`, then add `faiss` and `racplusplus` manually if needed.

## Data Expectations

`level-wise-rac.py` consumes a pickle file where each record maps a source ID to a list of embedding vectors:

```python
{
    "story-001": [np.ndarray(shape=(768,)), np.ndarray(shape=(768,))],
    "story-002": [np.ndarray(shape=(768,))]
}
```

Vectors are L2-normalised before clustering. The script also accepts an optional JSON file with pairwise similarity labels when building batches elsewhere in the repo; it is **not** required here.

## Usage

```bash
python clustering/level-wise-rac.py \
  --embedding-path data/embeddings.pkl \
  --output-path outputs/cluster_assignments.json \
  --thresholds 0.5,0.55,0.6 \
  --projections 384,192
```

Key options:

- `--thresholds`: Similarity thresholds (per level) passed to RAC++ (default `0.5,0.5,0.5`).
- `--projections`: Optional centroid projection dimensions for the levels *before* the next clustering step. Leave empty to automatically halve the dimensionality at each level.
- `--embedding-dim`: Expected dimensionality for embeddings (default `768`).
- `--rac-max-points`, `--rac-threads`, `--rac-metric`: Expose RAC++ runtime parameters.

The output JSON contains:

- `assignments`: A list of records, each describing the source ID, the embedding index, and the cluster label per level.
- `summary_by_id`: Counts of how many embeddings for each ID fall into each cluster at every level.
- `metadata`: Thresholds, projections, number of embeddings, and number of levels.

## Notes

- Thresholds are interpreted as cosine similarity cut-offs. RAC++ receives `(1 - threshold)` as epsilon.
- Projection dimensions must not exceed the original embedding dimensionality. The final level automatically skips projection.
- Large datasets may require tuning `--rac-max-points` and `--rac-threads` to balance recall and execution time.
