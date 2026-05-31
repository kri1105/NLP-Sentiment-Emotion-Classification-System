"""Prediction helpers used by the Streamlit app and any external caller."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import tensorflow as tf

from .model import EMOTION_LABELS
from .model_loader import LoadedModel
from .tokenizer import tokenize


EMOTION_INFO: Dict[int, Dict[str, str]] = {
    0: {"name": "Sadness", "emoji": "😢", "color": "#4A90E2"},
    1: {"name": "Joy", "emoji": "😊", "color": "#F5A623"},
    2: {"name": "Love", "emoji": "❤️", "color": "#E94B8B"},
    3: {"name": "Anger", "emoji": "😠", "color": "#D0021B"},
    4: {"name": "Fear", "emoji": "😨", "color": "#7B68EE"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#50E3C2"},
}


def predict_emotion(loaded: LoadedModel, text: str) -> np.ndarray:
    """Return the probability vector over the 6 emotion classes."""
    inputs = tokenize(loaded.tokenizer, [text], max_length=loaded.max_length)
    out = loaded.model(dict(inputs), training=False)
    out = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
    if loaded.returns_logits:
        out = tf.nn.softmax(out, axis=-1).numpy()
    return out[0]


def predict_top(loaded: LoadedModel, text: str) -> Dict[str, object]:
    probs = predict_emotion(loaded, text)
    idx = int(np.argmax(probs))
    return {
        "label": EMOTION_LABELS[idx],
        "label_idx": idx,
        "confidence": float(probs[idx]),
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(probs))},
    }
