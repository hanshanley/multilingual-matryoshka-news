"""AngIE embedding training script.

This refactored version mirrors the structure used in ``matryoshka-angie.py``:
configuration is supplied via the command line, data loading happens through
utility functions, and the training loop supports checkpointing and optional
validation. All absolute paths were replaced with user-provided arguments so
that the script can be published without sensitive defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class TrainingConfig:
    """Container for hyperparameters, paths, and runtime options."""

    train_path: Path
    output_dir: Path
    val_path: Optional[Path] = None
    model_name: str = "intfloat/multilingual-e5-base"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    batch_size: int = 16
    embedding_size: int = 768
    learning_rate: float = 2e-5
    epochs: int = 5
    validation_interval: int = 1_000
    val_ratio: float = 0.1
    random_seed: int = 42
    max_length: int = 512
    positive_margin_threshold: float = 0.75
    num_workers: int = 0
    use_label_sampler: bool = True

    def resolved_tokenizer(self) -> str:
        return self.tokenizer_name or self.model_name


class EmbeddingDataset(Dataset):
    """Pairs of texts with similarity margins for AngIE training."""

    def __init__(self, pairs: Sequence[Sequence], tokenizer: AutoTokenizer, max_length: int = 512):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Sequence:
        return self.pairs[idx]

    def _encode(self, texts: Iterable[str]) -> dict:
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

    def collate_fn(self, batch: Sequence[Sequence]) -> dict:
        text_a = [f"query: {pair[0]}" for pair in batch]
        text_b = [f"query: {pair[1]}" for pair in batch]
        margins = [float(pair[2]) for pair in batch]

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
    parser = argparse.ArgumentParser(description="Train AngIE multilingual embeddings.")
    parser.add_argument("--train-path", type=Path, required=True, help="Path to the training JSONL file.")
    parser.add_argument("--val-path", type=Path, default=None, help="Optional path to a separate validation JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and optimizer states.")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-base", help="Base model identifier.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer identifier (defaults to model name).")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--embedding-size", type=int, default=768, help="Hidden size used for projection truncation.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for AdamW.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--validation-interval", type=int, default=1_000, help="Validate every N update steps (<=0 disables intra-epoch validation).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Hold-out ratio when no validation set is provided.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for deterministic shuffling.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--positive-margin-threshold", type=float, default=0.75, help="Margins above this threshold are treated as positive (set to 1.0).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--disable-label-sampler", action="store_true", help="Use simple shuffled batching instead of the custom label-balanced sampler.")

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
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        max_length=args.max_length,
        positive_margin_threshold=args.positive_margin_threshold,
        num_workers=args.num_workers,
        use_label_sampler=not args.disable_label_sampler,
    )


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pairs(path: Path) -> List[Tuple[str, str, str, str, float]]:
    entries: List[Tuple[str, str, str, str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping line %s in %s due to JSON error: %s", line_num, path, exc)
                continue

            try:
                url_a, text_a, url_b, text_b = record[0], record[1], record[2], record[3]
                margin = float(record[-1])
            except (IndexError, TypeError, ValueError) as exc:
                logging.warning("Skipping malformed record at line %s in %s: %s", line_num, path, exc)
                continue

            entries.append((url_a, text_a, url_b, text_b, margin))

    logging.info("Loaded %s pairs from %s", len(entries), path)
    return entries


def assign_url_labels(entries: Sequence[Tuple[str, str, str, str, float]]) -> dict:
    graph: defaultdict[str, set[str]] = defaultdict(set)
    all_urls: set[str] = set()

    for url_a, _text_a, url_b, _text_b, margin in entries:
        all_urls.update({url_a, url_b})
        if margin > 0:
            graph[url_a].add(url_b)
            graph[url_b].add(url_a)

    for url in all_urls:
        graph.setdefault(url, set())

    labels: dict[str, int] = {}
    label_id = 0

    for url in sorted(all_urls):
        if url in labels:
            continue
        stack = [url]
        while stack:
            current = stack.pop()
            if current in labels:
                continue
            labels[current] = label_id
            stack.extend(sorted(neighbour for neighbour in graph[current] if neighbour not in labels))
        label_id += 1

    return labels


def prepare_dataset(entries: Sequence[Tuple[str, str, str, str, float]]) -> Tuple[List[List], List[List[int]]]:
    url_to_label = assign_url_labels(entries)
    dataset: List[List] = []
    label_pairs: List[List[int]] = []

    for url_a, text_a, url_b, text_b, margin in entries:
        dataset.append([text_a, text_b, float(margin)])
        label_pairs.append([url_to_label[url_a], url_to_label[url_b]])

    return dataset, label_pairs


def load_dataset_with_labels(path: Path) -> Tuple[List[List], List[List[int]]]:
    entries = read_pairs(path)
    return prepare_dataset(entries)


def split_train_val(
    dataset: List[List],
    labels: List[List[int]],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[List], List[List[int]], List[List], List[List[int]]]:
    if not dataset:
        return dataset, labels, [], []

    if val_ratio <= 0:
        return dataset, labels, [], []

    if val_ratio >= 1:
        raise ValueError("val_ratio must be in the range (0, 1) when no explicit validation path is provided")

    combined = list(zip(dataset, labels))
    rng = random.Random(seed)
    rng.shuffle(combined)

    split_index = max(1, int(len(combined) * (1 - val_ratio)))
    train_combined = combined[:split_index]
    val_combined = combined[split_index:]

    train_dataset, train_labels = zip(*train_combined) if train_combined else ([], [])
    val_dataset, val_labels = zip(*val_combined) if val_combined else ([], [])

    return list(train_dataset), list(train_labels), list(val_dataset), list(val_labels)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logging.info("CUDA unavailable; falling back to CPU.")
    return device


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_with_dropout(model: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    embeddings_diff = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    return embeddings, embeddings_diff


def contrastive_alignment_loss(
    embeddings_1: torch.Tensor,
    embeddings_1_diff: torch.Tensor,
    embeddings_2: torch.Tensor,
    embeddings_2_diff: torch.Tensor,
    *,
    device: torch.device,
    temperature: float = 0.05,
) -> torch.Tensor:
    data_full = torch.cat((embeddings_1, embeddings_2), dim=0)
    data_full_diff = torch.cat((embeddings_1_diff, embeddings_2_diff), dim=0)
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


def interleave_rows(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    return torch.stack((tensor_a, tensor_b), dim=1).reshape(-1, tensor_a.size(1))


def calculate_cosine_angle_loss(
    embeddings_1: torch.Tensor,
    embeddings_1_diff: torch.Tensor,
    embeddings_2: torch.Tensor,
    embeddings_2_diff: torch.Tensor,
    margins: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    combined = torch.cat(
        (
            interleave_rows(embeddings_1, embeddings_1_diff),
            interleave_rows(embeddings_2, embeddings_2_diff),
            interleave_rows(embeddings_1, embeddings_2),
            interleave_rows(embeddings_1_diff, embeddings_2_diff),
        ),
        dim=0,
    )

    batch_size = embeddings_1.size(0)
    positives = torch.ones(batch_size, device=device)
    labels = torch.cat((positives, positives, margins, margins), dim=0)

    pairwise_loss = cosine_loss(labels, combined) + angle_loss(labels, combined)
    alignment_loss = contrastive_alignment_loss(embeddings_1, embeddings_1_diff, embeddings_2, embeddings_2_diff, device=device)
    return pairwise_loss + alignment_loss


def forward_batch(model: AutoModel, batch: dict, device: torch.device, config: TrainingConfig) -> torch.Tensor:
    input_a = batch["text_pair1_token_ids"].to(device)
    mask_a = batch["text_pair1_attention_mask"].to(device)
    input_b = batch["text_pair2_token_ids"].to(device)
    mask_b = batch["text_pair2_attention_mask"].to(device)
    margins = batch["margins"].to(device)

    embeddings_a, embeddings_a_diff = encode_with_dropout(model, input_a, mask_a)
    embeddings_b, embeddings_b_diff = encode_with_dropout(model, input_b, mask_b)

    embedding_dim = min(config.embedding_size, embeddings_a.size(1))

    views = (
        F.normalize(embeddings_a[:, :embedding_dim], p=2, dim=1),
        F.normalize(embeddings_a_diff[:, :embedding_dim], p=2, dim=1),
        F.normalize(embeddings_b[:, :embedding_dim], p=2, dim=1),
        F.normalize(embeddings_b_diff[:, :embedding_dim], p=2, dim=1),
    )

    adjusted_margins = torch.where(
        margins >= config.positive_margin_threshold,
        torch.ones_like(margins),
        margins,
    )

    loss = calculate_cosine_angle_loss(*views, margins=adjusted_margins, device=device)
    batch_size = max(1, margins.size(0))
    return loss / batch_size


def evaluate(model: AutoModel, dataloader: DataLoader, device: torch.device, config: TrainingConfig) -> float:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model-epoch{epoch}-step{global_step}.pt"
    optimizer_path = output_dir / f"optimizer-epoch{epoch}-step{global_step}.pt"
    torch.save(model.state_dict(), model_path)
    torch.save({"epoch": epoch, "global_step": global_step, "optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    logging.info("Saved checkpoint to %s", model_path)


def build_dataloaders(
    config: TrainingConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dataset_raw, train_labels = load_dataset_with_labels(config.train_path)

    if config.val_path:
        val_dataset_raw, _ = load_dataset_with_labels(config.val_path)
    else:
        train_dataset_raw, train_labels, val_dataset_raw, _ = split_train_val(
            train_dataset_raw,
            train_labels,
            val_ratio=config.val_ratio,
            seed=config.random_seed,
        )

    train_dataset = EmbeddingDataset(train_dataset_raw, tokenizer, max_length=config.max_length)

    if config.use_label_sampler and train_labels:
        sampler = NonRepeatingBatchSampler(train_labels, config.batch_size, seed=config.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=train_dataset.collate_fn,
            num_workers=config.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=config.num_workers,
        )

    val_loader: Optional[DataLoader] = None
    if val_dataset_raw:
        val_dataset = EmbeddingDataset(val_dataset_raw, tokenizer, max_length=config.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=config.num_workers,
        )

    return train_loader, val_loader


def train(config: TrainingConfig) -> None:
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
