"""Generate embeddings for text inputs with Hugging Face encoders.

This module exposes a CLI that reads IDs and texts from JSON/JSONL/CSV files,
loads a transformer encoder, and exports a JSON dictionary mapping each ID to
its pooled embedding vector. The default configuration mirrors the
Matryoshka-style inference used elsewhere in the project (prefixing queries
when the model name contains ``e5``).

Example
-------

```bash
python embedding_training_and_inference/inference.py \
  --input-path data/texts.json \
  --output-path outputs/embeddings.json \
  --model-name intfloat/multilingual-e5-base
```

``data/texts.json`` should contain an object mapping IDs to raw text, or
alternatively JSONL/CSV files with ``id``/``text`` columns are also accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

try:  # Optional dependency for UMT5 encoders.
    from transformers import UMT5EncoderModel
except ImportError:  # pragma: no cover - only needed for specific checkpoints
    UMT5EncoderModel = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> List[Tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [(str(key), str(value)) for key, value in payload.items()]
    if isinstance(payload, list):
        records: List[Tuple[str, str]] = []
        for entry in payload:
            if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
                raise ValueError("JSON list records must contain 'id' and 'text' fields.")
            records.append((str(entry["id"]), str(entry["text"])))
        return records
    raise ValueError("JSON input must be an object (id->text) or a list of objects with 'id' and 'text'.")


def load_jsonl(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
            if isinstance(entry, dict) and "id" in entry and "text" in entry:
                records.append((str(entry["id"]), str(entry["text"])))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                records.append((str(entry[0]), str(entry[1])))
            else:
                raise ValueError(f"Line {line_num} must contain an object with 'id'/'text' or a two-element list.")
    return records


def load_csv(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV input must contain 'id' and 'text' columns.")
        for row in reader:
            records.append((str(row["id"]), str(row["text"])) )
    return records


def load_inputs(path: Path) -> List[Tuple[str, str]]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    raise ValueError("Supported input formats are JSON, JSONL, and CSV.")


# ---------------------------------------------------------------------------
# Dataset and batching
# ---------------------------------------------------------------------------

@dataclass
class TextRecord:
    identifier: str
    text: str


class TextDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[str, str]]):
        self._records = [TextRecord(identifier=identifier, text=text) for identifier, text in records]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def __getitem__(self, idx: int) -> TextRecord:
        return self._records[idx]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


def resolve_prefix(model_name: str, user_prefix: Optional[str]) -> str:
    if user_prefix is not None:
        return user_prefix
    return "query: " if "e5" in model_name.lower() else ""


def load_encoder(model_name: str, cache_dir: Optional[str]) -> Tuple[AutoTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    if model_name == "google/umt5-base":
        if UMT5EncoderModel is None:
            raise ImportError("Install transformers with UMT5 support to use google/umt5-base.")
        model = UMT5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    return tokenizer, model


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_fn_builder(tokenizer, max_length: int, prefix: str):
    def _collate(batch: Sequence[TextRecord]):
        ids = [record.identifier for record in batch]
        texts = [f"{prefix}{record.text}" for record in batch]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return ids, encodings

    return _collate


def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    dataset: TextDataset,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    prefix: str,
) -> Dict[str, List[float]]:
    collate_fn = collate_fn_builder(tokenizer, max_length=max_length, prefix=prefix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    results: Dict[str, List[float]] = {}
    with torch.no_grad():
        for ids, encodings in tqdm(dataloader, desc="Encoding", unit="batch"):
            encodings = {key: tensor.to(device) for key, tensor in encodings.items()}
            outputs = model(**encodings)
            pooled = mean_pooling(outputs, encodings["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for identifier, vector in zip(ids, pooled):
                results[identifier] = vector.cpu().tolist()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for text inputs using transformer encoders")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to JSON/JSONL/CSV file with IDs and texts.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON file for embeddings.")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-base", help="Hugging Face model identifier.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional cache directory for the tokenizer/model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--prefix", type=str, default=None, help="Optional text prefix applied before encoding (defaults to 'query: ' for e5 models).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_inputs(args.input_path)
    dataset = TextDataset(records)

    tokenizer, model = load_encoder(args.model_name, cache_dir=args.cache_dir)
    prefix = resolve_prefix(args.model_name, args.prefix)
    device = get_device(prefer_gpu=not args.cpu)

    embeddings = encode_texts(
        model,
        tokenizer,
        dataset,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prefix=prefix,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(embeddings, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
