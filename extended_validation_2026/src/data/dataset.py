"""
Data Loading for Diffusion-Native LM

Uses tiktoken (GPT-2 BPE, pure Rust, no C++ deps) as primary tokenizer.
Falls back to transformers only if tiktoken unavailable.

CRITICAL: Does NOT fall back to byte-level tokenization. If tokenization
fails, the script must fail loudly — byte tokenization invalidates the
experiments (vocab=257 instead of ~50K changes the task fundamentally).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path
import os


class TextDataset(Dataset):
    """Simple token-level dataset for autoregressive LM training."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _get_tokenizer():
    """
    Get a GPT-2 BPE tokenizer. Try tiktoken first (fast, no C++ deps).
    Fall back to transformers. NEVER fall back to bytes.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print(f"  Using tiktoken GPT-2 tokenizer (vocab={enc.n_vocab})")
        return ("tiktoken", enc)
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        print(f"  Using transformers GPT-2 tokenizer (vocab={tok.vocab_size})")
        return ("transformers", tok)
    except Exception as e:
        raise RuntimeError(
            f"No working GPT-2 tokenizer available. Error: {e}\n"
            "Install tiktoken: pip install tiktoken\n"
            "DO NOT fall back to byte-level tokenization — "
            "it invalidates the experiments."
        )


def _tokenize(text: str, tokenizer_info) -> list:
    kind, tok = tokenizer_info
    if kind == "tiktoken":
        return tok.encode(text, disallowed_special=())
    else:
        return tok.encode(text)


def load_wikitext(
    variant: str = "wikitext-103",
    split: str = "train",
    max_tokens: Optional[int] = None,
    cache_dir: str = "./token_cache",
) -> torch.Tensor:
    """
    Load and tokenize WikiText dataset. Caches tokenized output to disk
    so subsequent runs are instant.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_key = f"{variant}_{split}_{max_tokens or 'all'}.pt"
    cache_file = cache_path / cache_key

    if cache_file.exists():
        print(f"  Loading cached tokens from {cache_file}")
        return torch.load(cache_file)

    from datasets import load_dataset

    if variant == "wikitext-2":
        dataset_config = "wikitext-2-raw-v1"
    elif variant == "wikitext-103":
        dataset_config = "wikitext-103-raw-v1"
    else:
        raise ValueError(f"Unknown WikiText variant: {variant}")

    print(f"Loading {variant} ({split})...")
    dataset = load_dataset("wikitext", dataset_config, split=split)

    tokenizer_info = _get_tokenizer()

    all_tokens = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            tokens = _tokenize(text, tokenizer_info)
            all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                break

    if max_tokens:
        all_tokens = all_tokens[:max_tokens]

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  Tokenized {len(all_tokens):,} tokens, vocab_max={tokens_tensor.max().item()}")

    if tokens_tensor.max().item() < 1000:
        raise RuntimeError(
            f"Max token ID is {tokens_tensor.max().item()} — this looks like "
            "byte-level tokenization. BPE tokens should be up to ~50K."
        )

    torch.save(tokens_tensor, cache_file)
    print(f"  Cached to {cache_file}")

    return tokens_tensor


def load_slimpajama(
    split: str = "train",
    max_tokens: Optional[int] = None,
    cache_dir: str = "./token_cache",
) -> torch.Tensor:
    """
    Load and tokenize SlimPajama. This is a large dataset (627B tokens),
    so max_tokens is strongly recommended.

    SlimPajama is streamed by default to avoid downloading the full dataset.
    A max_tokens of 100M gives roughly WikiText-103 parity for comparison.

        Uses SlimPajama from HuggingFace. The default ID may change over time,
        so we try a small set of known IDs plus an optional override:
            - env SLIMPAJAMA_DATASET_ID (highest priority)
            - MBZUAI-LLM/SlimPajama-627B-DC
            - cerebras/SlimPajama-627B
            - cerebras/SlimPajama-627B-DC

        If access is gated/private in the current environment, provide a token
        via HF_TOKEN or HUGGINGFACE_HUB_TOKEN (or run `huggingface-cli login`).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_key = f"slimpajama_{split}_{max_tokens or 'all'}.pt"
    cache_file = cache_path / cache_key

    if cache_file.exists():
        print(f"  Loading cached tokens from {cache_file}")
        return torch.load(cache_file)

    from datasets import load_dataset

    print(f"Loading SlimPajama ({split}, streaming, max_tokens={max_tokens})...")

    candidate_ids = []
    override_id = os.environ.get("SLIMPAJAMA_DATASET_ID")
    if override_id:
        candidate_ids.append(override_id)
    candidate_ids.extend([
        "MBZUAI-LLM/SlimPajama-627B-DC",
        "cerebras/SlimPajama-627B",
        "cerebras/SlimPajama-627B-DC",
    ])

    # De-duplicate while preserving order
    candidate_ids = list(dict.fromkeys(candidate_ids))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    dataset = None
    load_errors = []
    for ds_id in candidate_ids:
        try:
            dataset = load_dataset(
                ds_id,
                split=split,
                streaming=True,
                token=hf_token,
            )
            print(f"  Using dataset source: {ds_id}")
            break
        except Exception as e:
            load_errors.append((ds_id, str(e)))

    if dataset is None:
        attempted = "\n".join([f"    - {ds}: {err}" for ds, err in load_errors])
        raise RuntimeError(
            "Failed to load SlimPajama from all known dataset IDs.\n"
            f"Tried:\n{attempted}\n\n"
            "Fix options:\n"
            "  1) Set SLIMPAJAMA_DATASET_ID to the correct Hub path for your environment\n"
            "  2) Authenticate for gated/private access:\n"
            "     - run `huggingface-cli login`, or\n"
            "     - export HF_TOKEN=<your_token>\n"
        )

    tokenizer_info = _get_tokenizer()

    all_tokens = []
    n_docs = 0
    for item in dataset:
        text = item.get("text", "").strip()
        if not text:
            continue
        tokens = _tokenize(text, tokenizer_info)
        all_tokens.extend(tokens)
        n_docs += 1

        if n_docs % 10000 == 0:
            print(f"  ... {n_docs:,} docs, {len(all_tokens):,} tokens")

        if max_tokens and len(all_tokens) >= max_tokens:
            break

    if max_tokens:
        all_tokens = all_tokens[:max_tokens]

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  Tokenized {len(all_tokens):,} tokens from {n_docs:,} docs, "
          f"vocab_max={tokens_tensor.max().item()}")

    if tokens_tensor.max().item() < 1000:
        raise RuntimeError(
            f"Max token ID is {tokens_tensor.max().item()} — byte-level tokenization detected."
        )

    torch.save(tokens_tensor, cache_file)
    print(f"  Cached to {cache_file}")

    return tokens_tensor


def _load_tokens(
    variant: str,
    split: str,
    max_tokens: Optional[int] = None,
    cache_dir: str = "./token_cache",
) -> torch.Tensor:
    """Dispatch to the right loader based on dataset variant."""
    if variant in ("wikitext-2", "wikitext-103"):
        return load_wikitext(variant, split, max_tokens, cache_dir)
    elif variant == "slimpajama":
        return load_slimpajama(split, max_tokens, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {variant}")


def create_dataloaders(
    variant: str = "wikitext-103",
    seq_len: int = 256,
    batch_size: int = 32,
    max_train_tokens: Optional[int] = None,
    num_workers: int = 4,
    cache_dir: str = "./token_cache",
) -> dict:
    """
    Create train/val/test dataloaders.

    For SlimPajama, which has no predefined val/test splits, we carve
    out the last 5% of training tokens as validation. SlimPajama is
    streamed, so max_train_tokens is strongly recommended (e.g. 100M
    for WikiText-103 parity).
    """
    train_tokens = _load_tokens(variant, "train", max_train_tokens, cache_dir)

    if variant == "slimpajama":
        # Carve validation from the end of training data
        n_val = min(len(train_tokens) // 20, 500_000)  # 5% or 500K, whichever is smaller
        val_tokens = train_tokens[-n_val:]
        test_tokens = val_tokens  # reuse for test
        train_tokens = train_tokens[:-n_val]
        print(f"  SlimPajama: carved {n_val:,} val tokens from training data")
    else:
        val_tokens = _load_tokens(variant, "validation", None, cache_dir)
        test_tokens = _load_tokens(variant, "test", None, cache_dir)

    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    test_dataset = TextDataset(test_tokens, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "test_tokens": len(test_tokens),
    }
