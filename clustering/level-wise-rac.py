"""Hierarchical clustering of Matryoshka-style embeddings with RAC++.

This module exposes a CLI that repeatedly applies the RAC++ clustering
algorithm across several abstraction levels. Each level clusters the current
set of vectors, aggregates their centroids, and optionally truncates the
centroid dimensionality before the next pass—mirroring the Matryoshka nesting
scheme. Connected cluster assignments are emitted for every original
embedding row, alongside a per-ID summary that shows how many embeddings fall
into each cluster at each level.

Example
-------

```bash
python clustering/level-wise-rac.py \
  --embedding-path data/matryoshka_embeddings.pkl \
  --output-path outputs/hierarchical_clusters.json \
  --thresholds 0.5,0.6,0.65
```

The pickle file must contain a mapping of `source_id -> List[np.ndarray]`
representing the embeddings to cluster. The script normalises vectors with
FAISS, executes RAC++, and stores assignments in JSON format.
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

try:  # pragma: no cover - soft dependency check
    import racplusplus
except ImportError as exc:  # pragma: no cover - fail fast if missing
    raise ImportError(
        "The 'racplusplus' package is required for hierarchical clustering. "
        "Install it from https://github.com/porterehunley/RACplusplus."
    ) from exc


@dataclass(frozen=True)
class LevelSetting:
    """Configuration for a single clustering level."""

    threshold: float
    centroid_projection: Optional[int] = None  # Dimensionality used for next level centroids

<<<<<<< HEAD
# Convert embeddings list to a NumPy array and normalize
embeddings = np.array(embeddings).reshape(len(embeddings), dim)
top_embeddings = np.array(embeddings)[,:192].reshape(len(embeddings), 192)
faiss.normalize_L2(embeddings)
=======
>>>>>>> 99cdfec (requirements)

@dataclass(frozen=True)
class RACConfig:
    """Wrapper for RAC++ runtime parameters."""

<<<<<<< HEAD
# Perform the first round of clustering using RAC++
labels = racplusplus.rac(top_embeddings, 1 - THRESHOLDS[0], None, 1000, 8, "cosine")
=======
    max_points: int = 1000
    n_threads: int = 8
    metric: str = "cosine"
>>>>>>> 99cdfec (requirements)


def parse_float_list(raw: str, *, argument: str) -> List[float]:
    """Parse a comma-separated string into floats."""

    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover - argparse should catch
        raise argparse.ArgumentTypeError(f"Invalid float sequence for {argument}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one value")
    return values


def parse_int_list(raw: str, *, argument: str) -> List[int]:
    """Parse a comma-separated string into integers."""

    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(f"Invalid integer sequence for {argument}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one value")
    return values


def load_embeddings(path: Path, expected_dim: Optional[int] = None) -> tuple[np.ndarray, List[str]]:
    """Load and L2-normalise embeddings from a pickle file."""

<<<<<<< HEAD
# Compute the average embedding for each cluster (second layer)
second_labels = []
for cluster in second_clusters_to_labels:
    new_clusters.append(np.array(np.average(embeddings[second_clusters_to_labels[cluster]], axis=0)[:768]))
    second_labels.append(cluster)
=======
    if not path.exists():  # pragma: no cover - guard for user error
        raise FileNotFoundError(f"Embedding pickle not found: {path}")
>>>>>>> 99cdfec (requirements)

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Embedding pickle must map source IDs to iterable embeddings.")

    vectors: List[np.ndarray] = []
    index_to_id: List[str] = []

    for source_id, embeddings in payload.items():
        if not isinstance(embeddings, Iterable):
            raise ValueError(f"Embeddings for ID {source_id!r} must be iterable.")
        for embedding in embeddings:
            array = np.asarray(embedding, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError(f"Embedding for ID {source_id!r} must be a 1D vector.")
            if expected_dim is not None and array.shape[0] != expected_dim:
                raise ValueError(
                    f"Embedding for ID {source_id!r} has dimension {array.shape[0]} "
                    f"but expected {expected_dim}."
                )
            vectors.append(array)
            index_to_id.append(str(source_id))

    if not vectors:
        raise ValueError("No embeddings were found in the supplied pickle.")

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix, index_to_id


def run_rac(vectors: np.ndarray, level: LevelSetting, rac_config: RACConfig) -> np.ndarray:
    """Execute RAC++ clustering for a single level and return cluster labels."""

    if vectors.size == 0:
        return np.empty((0,), dtype=np.int32)
    epsilon = 1.0 - level.threshold
    return racplusplus.rac(vectors, epsilon, None, rac_config.max_points, rac_config.n_threads, rac_config.metric)


def hierarchical_rac(
    embeddings: np.ndarray,
    level_settings: Sequence[LevelSetting],
    rac_config: RACConfig,
) -> List[np.ndarray]:
    """Run RAC++ across multiple hierarchical levels."""

    num_samples = embeddings.shape[0]
    if not level_settings:
        raise ValueError("At least one level setting is required.")

    vector_members: List[List[int]] = [[idx] for idx in range(num_samples)]
    current_vectors = embeddings
    assignments: List[np.ndarray] = []

    for level_idx, level in enumerate(level_settings):
        labels = run_rac(current_vectors, level, rac_config)
        cluster_to_members: MutableMapping[int, List[int]] = defaultdict(list)
        for vector_index, cluster_label in enumerate(labels):
            cluster_to_members[int(cluster_label)].extend(vector_members[vector_index])

        level_assignment = np.empty(num_samples, dtype=np.int64)
        for cluster_label, members in cluster_to_members.items():
            for member in members:
                level_assignment[member] = cluster_label
        assignments.append(level_assignment)

        is_last_level = level_idx == len(level_settings) - 1
        if is_last_level:
            break

        next_vectors: List[np.ndarray] = []
        next_vector_members: List[List[int]] = []
        for members in cluster_to_members.values():
            centroid = embeddings[members].mean(axis=0)
            if level.centroid_projection is not None:
                centroid = centroid[: level.centroid_projection]
            next_vectors.append(centroid.astype(np.float32, copy=False))
            next_vector_members.append(list(members))

        if not next_vectors:
            break

        current_vectors = np.vstack(next_vectors)
        vector_members = next_vector_members

    return assignments


def summarise_by_id(index_to_id: Sequence[str], assignments: Sequence[np.ndarray]) -> Dict[str, List[Dict[str, int]]]:
    """Aggregate assignment counts per source ID for each level."""

    summary: Dict[str, List[Counter]] = defaultdict(lambda: [Counter() for _ in assignments])
    for row_idx, source_id in enumerate(index_to_id):
        for level_idx, level_assignment in enumerate(assignments):
            summary[source_id][level_idx][int(level_assignment[row_idx])] += 1
    return {source_id: [dict(counter) for counter in counters] for source_id, counters in summary.items()}


def export_results(
    output_path: Path,
    index_to_id: Sequence[str],
    assignments: Sequence[np.ndarray],
    thresholds: Sequence[float],
    projections: Sequence[Optional[int]],
) -> None:
    """Write clustering assignments to disk as JSON."""

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
            "thresholds": thresholds,
            "centroid_projections": [proj if proj is not None else None for proj in projections],
        },
        "assignments": records,
        "summary_by_id": summarise_by_id(index_to_id, assignments),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_projections(embedding_dim: int, num_levels: int) -> List[int]:
    """Generate sensible default projections by halving the dimension each level."""

    projections: List[int] = []
    current_dim = embedding_dim
    for _ in range(max(0, num_levels - 1)):
        current_dim = max(1, current_dim // 2)
        projections.append(current_dim)
    return projections


def build_level_settings(
    thresholds: Sequence[float],
    projections: Sequence[Optional[int]],
    embedding_dim: int,
) -> List[LevelSetting]:
    """Combine thresholds and projections into a level configuration list."""

    if len(projections) < len(thresholds) - 1:
        raise ValueError("Provide at least len(thresholds) - 1 projection values.")

    settings: List[LevelSetting] = []
    for idx, threshold in enumerate(thresholds):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Thresholds must be in (0, 1]. Received {threshold}.")
        projection = projections[idx] if idx < len(projections) else None
        if projection is not None and (projection <= 0 or projection > embedding_dim):
            raise ValueError(f"Centroid projection must be between 1 and {embedding_dim}. Received {projection}.")
        settings.append(LevelSetting(threshold=threshold, centroid_projection=projection))
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical clustering of embeddings with RAC++")
    parser.add_argument("--embedding-path", type=Path, required=True, help="Pickle file containing ID -> embeddings mapping.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON file for cluster assignments.")
    parser.add_argument("--thresholds", type=str, default="0.5,0.5,0.5", help="Comma-separated similarity thresholds per level.")
    parser.add_argument("--projections", type=str, default="", help="Optional comma-separated centroid projection dimensions.")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Expected embedding dimensionality (used for validation).")
    parser.add_argument("--rac-max-points", type=int, default=1000, help="RAC++ max points parameter.")
    parser.add_argument("--rac-threads", type=int, default=8, help="Number of RAC++ worker threads.")
    parser.add_argument("--rac-metric", type=str, default="cosine", help="Similarity metric used by RAC++ (default: cosine).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds, argument="--thresholds")
    raw_projections = parse_int_list(args.projections, argument="--projections") if args.projections else None

    embeddings, index_to_id = load_embeddings(args.embedding_path, expected_dim=args.embedding_dim)
    embedding_dim = embeddings.shape[1]

    projections = (
        raw_projections
        if raw_projections is not None
        else default_projections(embedding_dim, len(thresholds))
    )
    if len(projections) < len(thresholds) - 1:
        defaults = default_projections(embedding_dim, len(thresholds))
        projections = projections + defaults[len(projections) : len(thresholds) - 1]
    level_settings = build_level_settings(thresholds, projections, embedding_dim)

    rac_config = RACConfig(max_points=args.rac_max_points, n_threads=args.rac_threads, metric=args.rac_metric)
    assignments = hierarchical_rac(embeddings, level_settings, rac_config)
    export_results(args.output_path, index_to_id, assignments, thresholds, [ls.centroid_projection for ls in level_settings])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
