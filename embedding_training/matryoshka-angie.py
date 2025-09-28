"""Matryoshka embedding training script.

This refactored version exposes configuration via CLI arguments, avoids
hard-coded file paths, and clarifies the training loop for publication.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    matryoshka_levels: Sequence[Tuple[float, float]] = (
        (0.25, 0.25),
        (0.50, 0.50),
        (0.75, 1.00),
    )

    def resolved_tokenizer(self) -> str:
        return self.tokenizer_name or self.model_name


class EmbeddingDataset(Dataset):
    """Pairs of texts with similarity margins for Matryoshka training."""

    def __init__(self, pairs: Sequence[Sequence], tokenizer: AutoTokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Sequence:
        sample = self.pairs[idx]
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            raise ValueError(
                "Each record must be an iterable with at least three elements (text_a, text_b, margin). "
                f"Got index {idx} -> {sample!r}"
            )
        return sample

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


def parse_args() -> TrainingConfig:
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
    )


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def load_jsonl(path: Path) -> List[Sequence]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[Sequence] = []
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
            records.append(record)

    logging.info("Loaded %s records from %s", len(records), path)
    return records


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logging.info("CUDA unavailable; falling back to CPU.")
    gc.collect()
    return device


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_with_dropout(model: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    embedding_diff = mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
    return embedding, embedding_diff


def binarize_margins(margins: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(margins >= threshold, torch.ones_like(margins), torch.zeros_like(margins))


def interleave_rows(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
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


def matryoshka_level_loss(
    emb1: torch.Tensor,
    emb1_diff: torch.Tensor,
    emb2: torch.Tensor,
    emb2_diff: torch.Tensor,
    *,
    binary_margins: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
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


def train(config: TrainingConfig) -> None:
    setup_logging()
    logging.info("Configuration: %s", config)

    train_records = load_jsonl(config.train_path)
    val_records = load_jsonl(config.val_path)

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.resolved_tokenizer(), cache_dir=config.cache_dir, use_fast=False)
    model = AutoModel.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    model.to(device)

    train_dataset = EmbeddingDataset(train_records, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader: Optional[DataLoader] = None
    if val_records:
        val_dataset = EmbeddingDataset(val_records, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

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
