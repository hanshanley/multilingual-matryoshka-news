# Embedding Training Utilities

Two executable scripts train multilingual encoders that respect Matryoshka- and AngIE-style objectives. Both rely on the same input format and reuse a connected-component label graph to avoid placing transitively similar news URLs into the same negative batch.

## Data Requirements

1. **Pairwise JSONL corpora** – Each line is a JSON array with five entries: `[url1, text1, url2, text2, similarity_score]`. The texts are fed to the encoder, and the `similarity_score` (e.g., SemEval 2022 Task 8 margins) guides loss weighting. NOTE: the zenodo file containing Jsonl currently has a VERY SIMILAR to VERY DISSIMILAR as the scores for clarity. These need to be changed to being 0.75 to 0 in intervals of 0.25 for you to run these files as is.
2. **Similarity label map** – A JSON dictionary (default `data/labels_semeval_2022_task_eight.json`) that maps each URL to a nested `{other_url: score}` object. Edges with scores ≥ `similarity_threshold` are treated as positive links when forming connected components.

Any URL missing from the label map is treated as its own singleton component, so it will not be considered a positive neighbour.

## Scripts

### `matryoshka-angie.py`
- Fine-tunes a backbone encoder (default `intfloat/multilingual-e5-base`) using Matryoshka multi-granularity losses.
- Builds batches with `NonRepeatingBatchSampler` so that batched pairs never include transitively similar URLs as artificial negatives.
- Checkpoints model and optimiser state whenever validation improves.

### `modified-angie.py`
- Implements the AngIE loss with cosine, angle, and contrastive objectives for comparison-style baselines.
- Shares the same batching strategy as the Matryoshka trainer. This should be used to train more traditional non-Matryoshka style embeddings. 

## Quick Start

```bash
python embedding_training/matryoshka-angie.py \
  --train-path data/train_pairs.jsonl \
  --val-path data/dev_pairs.jsonl \
  --output-dir outputs/matryoshka \
  --labels-path data/labels_semeval_2022_task_eight.json

python embedding_training/modified-angie.py \
  --train-path data/train_pairs.jsonl \
  --output-dir outputs/angie \
  --val-ratio 0.1
```

Both commands accept additional flags:

| Flag | Description |
| --- | --- |
| `--similarity-threshold` | Minimum score to treat two URLs as connected (default `0.25`). |
| `--batch-size` | Mini-batch size (default `16`). |
| `--epochs` | Number of training epochs (default `5`). |
| `--validation-interval` | In-epoch validation cadence; set ≤0 to disable. |
| `--disable-label-sampler` | Fall back to simple shuffling instead of the label-aware sampler. |
| `--max-length` | Maximum token length supplied to the tokenizer (default `512`). |
| `--random-seed` | Seed for shuffling and sampler reproducibility (default `42`). |

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support for GPU acceleration (optional but recommended)
- `transformers`, `tqdm`, and other dependencies listed in the project requirements

## Output

- Checkpoints and optimiser states are written to the directory specified by `--output-dir` with filenames `model-epoch{e}-step{s}.pt` and `optimizer-epoch{e}-step{s}.pt`.
- Training and validation losses are logged to STDOUT through `tqdm` progress bars and the standard logger.

Refer to the main repository README for evaluation scripts and downstream clustering pipelines.
- **Sampler rationale:** By default the scripts build batches with a `NonRepeatingBatchSampler` that references the similarity graph. This prevents URLs that are explicitly or transitively connected from being placed in the same mini-batch as artificial negatives. Disable it only if you need pure random sampling and can tolerate potential label leakage and for faster training. 
