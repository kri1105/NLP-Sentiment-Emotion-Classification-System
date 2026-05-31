"""Smart model loader that picks the best available checkpoint at runtime.

Search order:
    1. `model/best_emotion_model_roberta.h5` (new, higher-accuracy RoBERTa weights)
    2. `model/best_emotion_model.h5`         (legacy DistilBERT weights, 93%)

This means the app keeps working with the original weights right now, and
silently upgrades to the better model as soon as you drop the new checkpoint
into the `model/` directory after running the improved training notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel

from .model import EmotionClassifier, ImprovedBertClassifier, build_backbone
from .tokenizer import DEFAULT_MAX_LENGTH

ROBERTA_WEIGHTS = os.path.join("model", "best_emotion_model_roberta.h5")
LEGACY_WEIGHTS = os.path.join("model", "best_emotion_model.h5")


@dataclass
class LoadedModel:
    tokenizer: object
    model: tf.keras.Model
    backbone_name: str
    max_length: int
    returns_logits: bool


def _build_dummy(tokenizer, model):
    dummy = tokenizer("init", return_tensors="tf", padding="max_length", max_length=8)
    model(dict(dummy))


def _load_roberta() -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    backbone = build_backbone("roberta-base")
    backbone.trainable = True
    model = EmotionClassifier(backbone, return_logits=False)
    _build_dummy(tokenizer, model)
    model.load_weights(ROBERTA_WEIGHTS)
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        backbone_name="roberta-base",
        max_length=DEFAULT_MAX_LENGTH,
        returns_logits=False,
    )


def _load_legacy_distilbert() -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    model = ImprovedBertClassifier(bert)
    dummy = tokenizer("init", return_tensors="tf", padding=True)
    model(dict(dummy))
    model.load_weights(LEGACY_WEIGHTS)
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        backbone_name="distilbert-base-uncased",
        max_length=128,
        returns_logits=False,  # legacy model already softmax-activates
    )


def load_emotion_model() -> LoadedModel:
    """Return the best available emotion classifier."""
    if os.path.exists(ROBERTA_WEIGHTS):
        return _load_roberta()
    if os.path.exists(LEGACY_WEIGHTS):
        return _load_legacy_distilbert()
    raise FileNotFoundError(
        "No model weights found. Expected one of:\n"
        f"  - {ROBERTA_WEIGHTS} (improved, recommended)\n"
        f"  - {LEGACY_WEIGHTS} (legacy)\n"
        "Run the training notebook in notebook/ to generate weights."
    )
