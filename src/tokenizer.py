"""Tokenizer helpers used by both training and inference."""

from __future__ import annotations

from typing import Iterable, List

from transformers import AutoTokenizer

DEFAULT_MAX_LENGTH = 96  # 99% of dair-ai/emotion samples fit in 96 wordpieces


def load_tokenizer(model_name: str = "roberta-base"):
    """Load a fast tokenizer for the chosen backbone."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize(
    tokenizer,
    texts: Iterable[str],
    max_length: int = DEFAULT_MAX_LENGTH,
    return_tensors: str = "tf",
):
    """Pad-and-truncate tokenize a batch of texts."""
    if not isinstance(texts, list):
        texts = list(texts)
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )
