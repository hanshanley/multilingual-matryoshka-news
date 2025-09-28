"""Cluster Matryoshka embeddings hierarchically with RAC++.

The script clusters a collection of embeddings multiple times, gradually
expanding the dimensionality that is exposed to RAC++. The default schedule
matches the Matryoshka nesting pattern: first 192 dimensions, then 384, and
finally the full 768 dimensions. Each level aggregates centroids for the next
round so that cluster assignments remain consistent with the hierarchical
structure of the embeddings.

Example
-------

```bash
python clustering/level-wise-rac.py \
  --embedding-path data/embeddings.pkl \
  --output-path outputs/hierarchical_clusters.json
```

The pickled file must map source identifiers to lists of embedding vectors.
Outputs include per-embedding cluster assignments for every level and summary
statistics per source identifier.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import faiss
import numpy as np

try:  # pragma: no cover - optional dependency guard
    import racplusplus
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'racplusplus' package is required. Install it from "
        "https://github.com/porterehunley/RACplusplus"
    ) from exc


@dataclass(frozen=True)
class RACConfig:
    max_points: int = 1000
    n_threads: int = 8
    metric: str = "cosine"


@dataclass(frozen=True)
class LevelSetting:
    threshold: float
    dimension: int


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_embeddings(path: Path) -> tuple[np.ndarray, List[str]]:
    """Load a mapping of ID -> embeddings and return stacked vectors."""

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Embedding pickle must map source IDs to iterables of embeddings.")

    vectors: List[np.ndarray] = []
    index_to_id: List[str] = []

    for source_id, embeddings in payload.items():
        if not isinstance(embeddings, Iterable):
            raise ValueError(f"Embeddings for ID {source_id!r} must be iterable.")
        for embedding in embeddings:
            array = np.asarray(embedding, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError(f"Embedding for ID {source_id!r} must be a 1D vector.")
            vectors.append(array)
            index_to_id.append(str(source_id))

    if not vectors:
        raise ValueError("No embeddings found in the supplied pickle.")

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix, index_to_id


# ---------------------------------------------------------------------------
# Hierarchical clustering logic
# ---------------------------------------------------------------------------

def compute_level_vectors(
    embeddings: np.ndarray,
    vector_members: Sequence[Sequence[int]],
    dimension: int,
) -> np.ndarray:
    """Compute level-specific vectors by averaging truncated embeddings."""

    truncated_dim = min(dimension, embeddings.shape[1])
    level_vectors: List[np.ndarray] = []
    for members in vector_members:
        subset = embeddings[np.asarray(members), :truncated_dim]
        centroid = subset.mean(axis=0)
        level_vectors.append(centroid.astype(np.float32))

    matrix = np.vstack(level_vectors)
    faiss.normalize_L2(matrix)
    return matrix


def run_rac(vectors: np.ndarray, threshold: float, rac_config: RACConfig) -> np.ndarray:
    """Run RAC++ on the provided vectors and return integer cluster labels."""

    epsilon = 1.0 - threshold
    return racplusplus.rac(vectors, epsilon, None, rac_config.max_points, rac_config.n_threads, rac_config.metric)


def hierarchical_cluster(
    embeddings: np.ndarray,
    settings: Sequence[LevelSetting],
    rac_config: RACConfig,
) -> List[np.ndarray]:
    """Perform hierarchical clustering using the supplied level settings."""

    num_samples = embeddings.shape[0]
    if not settings:
        raise ValueError("At least one level setting must be provided.")

    vector_members: List[List[int]] = [[idx] for idx in range(num_samples)]
    assignments: List[np.ndarray] = []

    for level in settings:
        level_vectors = compute_level_vectors(embeddings, vector_members, level.dimension)
        labels = run_rac(level_vectors, level.threshold, rac_config)

        cluster_to_members: MutableMapping[int, List[int]] = defaultdict(list)
        for vec_idx, cluster_label in enumerate(labels):
            cluster_to_members[int(cluster_label)].extend(vector_members[vec_idx])

        level_assignment = np.empty(num_samples, dtype=np.int64)
        for cluster_label, members in cluster_to_members.items():
            level_assignment[np.asarray(members)] = cluster_label
        assignments.append(level_assignment)

        vector_members = [members for members in cluster_to_members.values()]

    return assignments


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def summarise_by_id(index_to_id: Sequence[str], assignments: Sequence[np.ndarray]) -> Dict[str, List[Dict[str, int]]]:
    """Aggregate cluster counts per source ID for each level."""

    summary: Dict[str, List[Counter]] = defaultdict(lambda: [Counter() for _ in assignments])
    for idx, source_id in enumerate(index_to_id):
        for level_idx, level_assignment in enumerate(assignments):
            summary[source_id][level_idx][int(level_assignment[idx])] += 1
    return {source_id: [dict(counter) for counter in counters] for source_id, counters in summary.items()}


def export_assignments(
    output_path: Path,
    index_to_id: Sequence[str],
    assignments: Sequence[np.ndarray],
    settings: Sequence[LevelSetting],
) -> None:
    """Persist clustering results and metadata in JSON format."""

    records = [
        {
            "embedding_index": idx,
            "source_id": source_id,
            "levels": [int(level_assignment[idx]) for level_assignment in assignments],
        }
        for idx, source_id in enumerate(index_to_id)
    ]

    payload = {
        "metadata": {
            "num_embeddings": len(index_to_id),
            "num_levels": len(assignments),
            "thresholds": [setting.threshold for setting in settings],
            "dimensions": [setting.dimension for setting in settings],
        },
        "assignments": records,
        "summary_by_id": summarise_by_id(index_to_id, assignments),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Command Line 
# ---------------------------------------------------------------------------

def parse_float_sequence(raw: str, argument: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(f"Invalid float sequence for {argument}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one value")
    return values


def parse_int_sequence(raw: str, argument: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(f"Invalid integer sequence for {argument}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical clustering of embeddings with RAC++")
    parser.add_argument("--embedding-path", type=Path, required=True, help="Pickle file containing ID -> embeddings mapping.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON file for cluster assignments.")
    parser.add_argument("--thresholds", type=str, default="0.5,0.55,0.6", help="Comma-separated similarity thresholds per level.")
    parser.add_argument(
        "--dimensions",
        type=str,
        default="192,384,768",
        help="Comma-separated dimensionality schedule (defaults to Matryoshka 192/384/768).",
    )
    parser.add_argument("--rac-max-points", type=int, default=1000, help="RAC++ max points parameter.")
    parser.add_argument("--rac-threads", type=int, default=8, help="Number of RAC++ worker threads.")
    parser.add_argument("--rac-metric", type=str, default="cosine", help="Similarity metric used by RAC++ (default: cosine).")
    return parser.parse_args()


def build_level_settings(thresholds: Sequence[float], dimensions: Sequence[int]) -> List[LevelSetting]:
    if len(thresholds) != len(dimensions):
        raise ValueError("Provide the same number of thresholds and dimensions.")
    settings: List[LevelSetting] = []
    for threshold, dimension in zip(thresholds, dimensions):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Thresholds must be in (0, 1]. Received {threshold}.")
        if dimension <= 0:
            raise ValueError(f"Dimensions must be positive integers. Received {dimension}.")
        settings.append(LevelSetting(threshold=threshold, dimension=dimension))
    return settings


def main() -> None:
    args = parse_args()
    thresholds = parse_float_sequence(args.thresholds, argument="--thresholds")
    dimensions = parse_int_sequence(args.dimensions, argument="--dimensions")
    settings = build_level_settings(thresholds, dimensions)

    embeddings, index_to_id = load_embeddings(args.embedding_path)
    rac_config = RACConfig(max_points=args.rac_max_points, n_threads=args.rac_threads, metric=args.rac_metric)

    assignments = hierarchical_cluster(embeddings, settings, rac_config)
    export_assignments(args.output_path, index_to_id, assignments, settings)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
