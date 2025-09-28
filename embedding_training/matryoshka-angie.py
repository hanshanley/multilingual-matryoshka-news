"""Train Matryoshka-style multilingual embeddings with configurable CLI options.

Run ``python matryoshka-angie.py --train-path <train.jsonl> --val-path <val.jsonl> --output-dir <ckpt_dir>``
to fine-tune a backbone encoder using multi-resolution Matryoshka losses.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class TrainingConfig:
    """Container for hyperparameters and paths used during training."""

    train_path: Path
    val_path: Path
    output_dir: Path
    model_name: str = "intfloat/multilingual-e5-base"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    batch_size: int = 16
    embedding_size: int = 768
    learning_rate: float = 2e-5
    epochs: int = 5
    validation_interval: int = 10_000
    labels_path: Optional[Path] = None
    similarity_threshold: float = 0.25
    random_seed: int = 42
    max_length: int = 512
    num_workers: int = 0
    use_label_sampler: bool = True
    matryoshka_levels: Sequence[Tuple[float, float]] = (
        (0.25, 0.25),
        (0.50, 0.50),
        (0.75, 1.00),
    )

    def resolved_tokenizer(self) -> str:
        return self.tokenizer_name or self.model_name


class EmbeddingDataset(Dataset):
    """Pairs of texts with similarity margins for Matryoshka training."""

    def __init__(self, pairs: Sequence[Mapping[str, object]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        return self.pairs[idx]

    def _encode(self, texts: Iterable[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

    def collate_fn(self, batch: Sequence[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
        text_a = [f"query: {pair['text_a']}" for pair in batch]
        text_b = [f"query: {pair['text_b']}" for pair in batch]
        margins = [float(pair['margin']) for pair in batch]

        encoded_a = self._encode(text_a)
        encoded_b = self._encode(text_b)

        return {
            "text_pair1_token_ids": encoded_a["input_ids"].long(),
            "text_pair1_attention_mask": encoded_a["attention_mask"].long(),
            "text_pair2_token_ids": encoded_b["input_ids"].long(),
            "text_pair2_attention_mask": encoded_b["attention_mask"].long(),
            "margins": torch.tensor(margins, dtype=torch.float32),
        }


class NonRepeatingBatchSampler(Sampler[List[int]]):
    """Sample batches without repeating label IDs inside a mini-batch."""

    def __init__(self, labels: Sequence[Sequence[int]], batch_size: int, seed: int = 42):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.labels = list(labels)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        if not self.labels:
            return 0
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        if not self.labels:
            return

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        indices = list(range(len(self.labels)))
        rng.shuffle(indices)

        def label_pair(idx: int) -> Tuple[int, int]:
            label = self.labels[idx]
            if not isinstance(label, (list, tuple)) or len(label) < 2:
                raise ValueError(f"Each label entry must contain at least two IDs. Got: {label!r}")
            return int(label[0]), int(label[1])

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_labels = set()
        leftovers: List[int] = []

        for idx in indices:
            a, b = label_pair(idx)
            if a not in current_labels and b not in current_labels:
                current_batch.append(idx)
                current_labels.update({a, b})
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_labels.clear()
            else:
                leftovers.append(idx)

        attempts = 0
        max_attempts = len(leftovers)
        while leftovers and attempts < max_attempts:
            idx = leftovers.pop(0)
            a, b = label_pair(idx)
            if a not in current_labels and b not in current_labels:
                current_batch.append(idx)
                current_labels.update({a, b})
            else:
                leftovers.append(idx)

            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_labels.clear()

            attempts += 1

        if current_batch:
            batches.append(current_batch)

        rng.shuffle(batches)
        for batch in batches:
            yield batch


def parse_args() -> TrainingConfig:
    """Parse CLI arguments and convert them into a ``TrainingConfig`` instance."""
    parser = argparse.ArgumentParser(description="Train Matryoshka multilingual embeddings.")
    parser.add_argument("--train-path", type=Path, required=True, help="Path to the training JSONL file.")
    parser.add_argument("--val-path", type=Path, required=True, help="Path to the validation JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints.")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-base", help="Base model name on Hugging Face.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name (defaults to model name).")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional cache directory for Hugging Face downloads.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--embedding-size", type=int, default=768, help="Size of the base embedding.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for AdamW.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10_000,
        help="Validate every N update steps (set <=0 to skip intermediate validation).",
    )
    parser.add_argument("--labels-path", type=Path, default=None, help="Optional JSON file containing URL similarity labels.")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="Edge weight >= threshold counts as similar.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for deterministic shuffling.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--disable-label-sampler", action="store_true", help="Use shuffled batches instead of label-aware sampler.")

    args = parser.parse_args()
    return TrainingConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        embedding_size=args.embedding_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        validation_interval=args.validation_interval,
        labels_path=args.labels_path,
        similarity_threshold=args.similarity_threshold,
        random_seed=args.random_seed,
        max_length=args.max_length,
        num_workers=args.num_workers,
        use_label_sampler=not args.disable_label_sampler,
    )


def setup_logging() -> None:
    """Initialise a lightweight console logger."""
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def set_random_seed(seed: int) -> None:
    """Seed Python, PyTorch, and CUDA RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_pair_record(record: object) -> Dict[str, object]:
    """Normalise heterogeneous pair records into a standard dictionary structure."""
    if isinstance(record, MutableMapping):
        url_a = record.get("url_a") or record.get("url1")
        text_a = record.get("text_a") or record.get("text1") or record.get("data1")
        url_b = record.get("url_b") or record.get("url2")
        text_b = record.get("text_b") or record.get("text2") or record.get("data2")
        margin = record.get("margin") or record.get("similarity") or record.get("score")
    elif isinstance(record, (list, tuple)):
        if len(record) >= 5:
            url_a, text_a, url_b, text_b, margin = record[:5]
        elif len(record) == 3:
            url_a = url_b = None
            text_a, text_b, margin = record
        else:
            raise ValueError(f"Expected 3 or 5 elements, got {len(record)}")
    else:
        raise TypeError(f"Unsupported record type: {type(record)}")

    if text_a is None or text_b is None:
        raise ValueError("Both text_a and text_b must be present in each record.")
    if margin is None:
        raise ValueError("Each record must include a similarity score or margin.")

    return {
        "url_a": url_a,
        "text_a": text_a,
        "url_b": url_b,
        "text_b": text_b,
        "margin": float(margin),
    }


def load_similarity_labels(path: Path) -> Dict[str, Dict[str, float]]:
    """Load a nested similarity map from JSON for downstream clustering."""
    if not path or not path.exists():
        raise FileNotFoundError(f"Similarity label file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, MutableMapping):
        raise ValueError("Similarity labels must be a JSON object mapping URLs to score dictionaries.")

    normalised: Dict[str, Dict[str, float]] = {}
    for url, neighbours in data.items():
        if not isinstance(neighbours, MutableMapping):
            raise ValueError(f"Expected nested mappings for URL {url}, got {type(neighbours)}")
        normalised[url] = {str(other): float(score) for other, score in neighbours.items()}
    return normalised


def _connected_components(graph: Mapping[str, Set[str]]) -> List[Set[str]]:
    """Compute connected components for an undirected similarity graph."""
    components: List[Set[str]] = []
    visited: Set[str] = set()

    for node in sorted(graph.keys()):
        if node in visited:
            continue
        stack = [node]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(graph[current] - visited))
        components.append(component)

    return components


def derive_similarity_components(
    similarity_map: Mapping[str, Mapping[str, float]],
    threshold: float = 0.25,
) -> List[Set[str]]:
    """Compute connected components where edges meet the similarity threshold."""
    graph: MutableMapping[str, Set[str]] = defaultdict(set)
    for url, neighbours in similarity_map.items():
        graph.setdefault(url, set())
        for neighbour, score in neighbours.items():
            graph.setdefault(neighbour, set())
            if float(score) >= threshold:
                graph[url].add(neighbour)
                graph[neighbour].add(url)
    return _connected_components(graph)


def build_label_lookup(
    similarity_map: Optional[Mapping[str, Mapping[str, float]]] = None,
    records: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    threshold: float = 0.25,
) -> Dict[str, int]:
    """Assign consecutive cluster identifiers to URLs using explicit labels and record margins."""
    graph: MutableMapping[str, Set[str]] = defaultdict(set)

    if similarity_map:
        for url, neighbours in similarity_map.items():
            graph.setdefault(url, set())
            for neighbour, score in neighbours.items():
                graph.setdefault(neighbour, set())
                if float(score) >= threshold:
                    graph[url].add(neighbour)
                    graph[neighbour].add(url)

    if records:
        for record in records:
            url_a = record.get("url_a")
            url_b = record.get("url_b")
            margin = record.get("margin")
            if url_a is not None:
                graph.setdefault(url_a, set())
            if url_b is not None:
                graph.setdefault(url_b, set())
            if (
                url_a is not None
                and url_b is not None
                and margin is not None
                and float(margin) >= threshold
            ):
                graph[url_a].add(url_b)
                graph[url_b].add(url_a)

    if not graph:
        return {}

    components = _connected_components(graph)
    label_lookup: Dict[str, int] = {}
    for idx, component in enumerate(components):
        for url in component:
            label_lookup[url] = idx
    return label_lookup


def load_jsonl(path: Path) -> List[Mapping[str, object]]:
    """Load a JSONL file containing training pairs and normalise each record."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping line %s in %s due to JSON error: %s", line_num, path, exc)
                continue
            try:
                normalised = normalize_pair_record(record)
            except (TypeError, ValueError) as exc:
                logging.warning("Skipping line %s in %s due to format error: %s", line_num, path, exc)
                continue
            records.append(normalised)

    logging.info("Loaded %s records from %s", len(records), path)
    return records


def get_device() -> torch.device:
    """Select CUDA when available, otherwise default to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logging.info("CUDA unavailable; falling back to CPU.")
    gc.collect()
    return device


def build_dataloaders(
    config: TrainingConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train/validation dataloaders with optional label-aware batching."""
    train_records = load_jsonl(config.train_path)
    val_records: List[Mapping[str, object]] = []
    if config.val_path:
        val_records = load_jsonl(config.val_path)

    similarity_map = load_similarity_labels(config.labels_path) if config.labels_path else None
    combined_records = train_records + val_records
    label_lookup = build_label_lookup(similarity_map, records=combined_records, threshold=config.similarity_threshold)

    train_dataset = EmbeddingDataset(train_records, tokenizer, max_length=config.max_length)

    can_use_sampler = (
        config.use_label_sampler
        and train_records
        and all(record.get("url_a") and record.get("url_b") for record in train_records)
        and label_lookup
    )

    train_loader: DataLoader
    if can_use_sampler:
        try:
            train_labels = [
                (label_lookup[record["url_a"]], label_lookup[record["url_b"]])
                for record in train_records
            ]
        except KeyError:
            can_use_sampler = False
        else:
            sampler = NonRepeatingBatchSampler(train_labels, config.batch_size, seed=config.random_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                collate_fn=train_dataset.collate_fn,
                num_workers=config.num_workers,
            )
    if not can_use_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=config.num_workers,
        )

    val_loader: Optional[DataLoader] = None
    if val_records:
        val_dataset = EmbeddingDataset(val_records, tokenizer, max_length=config.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=config.num_workers,
        )

    return train_loader, val_loader


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling with an attention mask to obtain sentence embeddings."""
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_with_dropout(model: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two forward passes that reflect dropout variability for contrastive losses."""
    embedding = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    embedding_diff = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    return embedding, embedding_diff


def binarize_margins(margins: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert continuous margins into binary labels based on a threshold."""
    return torch.where(margins >= threshold, torch.ones_like(margins), torch.zeros_like(margins))


def interleave_rows(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """Interleave rows of two tensors so paired embeddings appear consecutively."""
    return torch.stack((tensor_a, tensor_b), dim=1).reshape(-1, tensor_a.size(1))


def contrastive_loss(
    embeddings_1: torch.Tensor,
    embeddings_1_diff: torch.Tensor,
    embeddings_2: torch.Tensor,
    embeddings_2_diff: torch.Tensor,
    *,
    device: torch.device,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE-style contrastive term linking paired dropout views."""
    data_full = torch.cat((embeddings_1, embeddings_2_diff), dim=0)
    data_full_diff = torch.cat((embeddings_1_diff, embeddings_2), dim=0)
    similarity = torch.mm(data_full, data_full_diff.t()) / temperature

    batch_size = embeddings_1.size(0)
    indices = torch.arange(batch_size, device=device)

    mask = torch.zeros_like(similarity, dtype=torch.bool, device=device)
    mask[indices, indices] = True
    mask[indices, indices + batch_size] = True
    mask[indices + batch_size, indices] = True
    mask[indices + batch_size, indices + batch_size] = True

    numerator = torch.exp(similarity) * mask.float()
    numerator = numerator.sum(dim=1)
    denominator = torch.exp(similarity).sum(dim=1)
    return -torch.log(numerator / denominator).sum()


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """Ranking-friendly cosine loss adapted from the CoSENT formulation."""
    order_matrix = (y_true[:, None] < y_true[None, :]).float()
    normalized = F.normalize(y_pred, p=2, dim=1)
    similarities = torch.sum(normalized[::2] * normalized[1::2], dim=1) * tau
    differences = similarities[:, None] - similarities[None, :]
    differences = (differences - (1 - order_matrix) * 1e12).reshape(-1)
    zero = torch.tensor([0.0], device=differences.device)
    differences = torch.cat((zero, differences), dim=0)
    return torch.logsumexp(differences, dim=0)


def angle_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 1.0,
    pooling_strategy: str = "sum",
) -> torch.Tensor:
    """Phase-aware loss borrowed from AnglE that complements cosine ranking."""
    order_matrix = (y_true[:, None] < y_true[None, :]).float()

    real_part, imag_part = torch.chunk(y_pred, 2, dim=1)
    a, b = real_part[::2], imag_part[::2]
    c, d = real_part[1::2], imag_part[1::2]

    denominator = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    real = (a * c + b * d) / denominator
    imag = (b * c - a * d) / denominator

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True).sqrt()
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True).sqrt()
    scale = dz / dw
    real /= scale
    imag /= scale

    pooled = torch.cat((real, imag), dim=1)
    if pooling_strategy == "sum":
        pooled = torch.sum(pooled, dim=1)
    elif pooling_strategy == "mean":
        pooled = torch.mean(pooled, dim=1)
    else:
        raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

    pooled = torch.abs(pooled) * tau
    differences = pooled[:, None] - pooled[None, :]
    differences = (differences - (1 - order_matrix) * 1e12).reshape(-1)
    zero = torch.tensor([0.0], device=differences.device)
    differences = torch.cat((zero, differences), dim=0)
    return torch.logsumexp(differences, dim=0)


def matryoshka_level_loss(
    emb1: torch.Tensor,
    emb1_diff: torch.Tensor,
    emb2: torch.Tensor,
    emb2_diff: torch.Tensor,
    *,
    binary_margins: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Aggregate Matryoshka losses for one resolution slice."""
    combined = torch.cat(
        (
            interleave_rows(emb1, emb1_diff),
            interleave_rows(emb2, emb2_diff),
            interleave_rows(emb1, emb2),
            interleave_rows(emb1_diff, emb2_diff),
            interleave_rows(emb1, emb2_diff),
            interleave_rows(emb1_diff, emb2),
        ),
        dim=0,
    )

    batch_size = emb1.size(0)
    positives = torch.ones(batch_size, device=device)
    labels = torch.cat(
        (
            positives,
            positives,
            binary_margins,
            binary_margins,
            binary_margins,
            binary_margins,
        ),
        dim=0,
    )

    pairwise_loss = cosine_loss(labels, combined) + angle_loss(labels, combined)
    contrastive = contrastive_loss(emb1, emb1_diff, emb2, emb2_diff, device=device)
    return pairwise_loss + contrastive


def forward_batch(model: AutoModel, batch: dict, device: torch.device, config: TrainingConfig) -> torch.Tensor:
    """Run a training step over one mini-batch and aggregate Matryoshka losses."""
    input_a = batch["text_pair1_token_ids"].to(device)
    mask_a = batch["text_pair1_attention_mask"].to(device)
    input_b = batch["text_pair2_token_ids"].to(device)
    mask_b = batch["text_pair2_attention_mask"].to(device)
    margins = batch["margins"].to(device)

    emb_a, emb_a_diff = encode_with_dropout(model, input_a, mask_a)
    emb_b, emb_b_diff = encode_with_dropout(model, input_b, mask_b)

    total_loss = torch.tensor(0.0, device=device)
    for threshold, ratio in config.matryoshka_levels:
        embedding_dim = max(1, int(config.embedding_size * ratio))
        views = (
            F.normalize(emb_a[:, :embedding_dim], p=2, dim=1),
            F.normalize(emb_a_diff[:, :embedding_dim], p=2, dim=1),
            F.normalize(emb_b[:, :embedding_dim], p=2, dim=1),
            F.normalize(emb_b_diff[:, :embedding_dim], p=2, dim=1),
        )
        binary_margins = binarize_margins(margins, threshold)
        level_loss = matryoshka_level_loss(*views, binary_margins=binary_margins, device=device)
        total_loss += level_loss

    return total_loss / config.batch_size


def evaluate(model: AutoModel, dataloader: DataLoader, device: torch.device, config: TrainingConfig) -> float:
    """Compute the average training loss across a validation dataloader."""
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            batch_loss = forward_batch(model, batch, device, config)
            losses.append(batch_loss.item())
    return float(sum(losses) / len(losses)) if losses else 0.0


def save_checkpoint(
    model: AutoModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    output_dir: Path,
) -> None:
    """Persist model and optimiser state so training can be resumed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model-epoch{epoch}-step{global_step}.pt"
    optimizer_path = output_dir / f"optimizer-epoch{epoch}-step{global_step}.pt"
    torch.save(model.state_dict(), model_path)
    torch.save({"epoch": epoch, "global_step": global_step, "optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    logging.info("Saved checkpoint to %s", model_path)


def train(config: TrainingConfig) -> None:
    """Full training entry point orchestrating loading, optimisation, and validation."""
    setup_logging()
    logging.info("Configuration: %s", config)
    set_random_seed(config.random_seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.resolved_tokenizer(), cache_dir=config.cache_dir, use_fast=False)
    model = AutoModel.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    model.to(device)

    train_loader, val_loader = build_dataloaders(config, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"train-{epoch}")

        for batch in progress:
            optimizer.zero_grad()
            batch_loss = forward_batch(model, batch, device, config)
            batch_loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += batch_loss.item()
            progress.set_postfix({"loss": batch_loss.item()})

            if (
                val_loader is not None
                and config.validation_interval > 0
                and global_step % config.validation_interval == 0
            ):
                val_loss = evaluate(model, val_loader, device, config)
                logging.info("Step %s | validation loss: %.5f", global_step, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, global_step, config.output_dir)
                model.train()

        epoch_loss = running_loss / max(1, len(train_loader))
        logging.info("Epoch %s | training loss: %.5f", epoch, epoch_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, config)
            logging.info("Epoch %s | validation loss: %.5f", epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, global_step, config.output_dir)

    logging.info("Training complete. Best validation loss: %.5f", best_val_loss)


if __name__ == "__main__":
    train(parse_args())
