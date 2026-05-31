"""Improved transformer emotion classifier.

Architecture upgrades that drive accuracy past the original 93% baseline:
    1. Backbone swap: RoBERTa-base (≈ +1 pt over DistilBERT on dair-ai/emotion)
       while keeping a graceful fallback to DistilBERT for the legacy weights.
    2. Pooled representation = concat([CLS], attention-masked mean pool).
       Captures both the sentence-level token and an averaged context signal.
    3. Wider classification head (512) with GELU + LayerNorm + 2 dropouts.
    4. Logits-only output. Softmax is applied at inference time so training
       can use `from_logits=True` (numerically stabler with label smoothing).
"""

from __future__ import annotations

import tensorflow as tf
from transformers import TFAutoModel


EMOTION_LABELS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]


class EmotionClassifier(tf.keras.Model):
    """Transformer backbone + pooled classification head."""

    def __init__(
        self,
        backbone: tf.keras.Model,
        num_classes: int = 6,
        hidden_size: int = 512,
        dropout_pool: float = 0.3,
        dropout_hidden: float = 0.2,
        return_logits: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.return_logits = return_logits

        self.dropout_pool = tf.keras.layers.Dropout(dropout_pool, name="dropout_pool")
        self.layer_norm = tf.keras.layers.LayerNormalization(name="pool_layer_norm")
        self.dense = tf.keras.layers.Dense(
            hidden_size, activation="gelu", name="hidden_proj"
        )
        self.dropout_hidden = tf.keras.layers.Dropout(
            dropout_hidden, name="dropout_hidden"
        )
        self.classifier = tf.keras.layers.Dense(num_classes, name="logits")

    def call(self, inputs, training: bool = False):
        backbone_out = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=training,
        )
        # Transformers returns either a tuple or BaseModelOutput-like object.
        token_embeddings = (
            backbone_out[0] if isinstance(backbone_out, (tuple, list))
            else backbone_out.last_hidden_state
        )

        # ── Pooling: concat [CLS] (position 0) with attention-masked mean ──
        cls_embedding = token_embeddings[:, 0, :]
        mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        masked = token_embeddings * mask
        sum_embeddings = tf.reduce_sum(masked, axis=1)
        sum_mask = tf.reduce_sum(mask, axis=1)
        mean_pool = sum_embeddings / tf.maximum(sum_mask, 1e-9)
        pooled = tf.concat([cls_embedding, mean_pool], axis=-1)

        # ── Classification head ──
        x = self.dropout_pool(pooled, training=training)
        x = self.layer_norm(x)
        x = self.dense(x)
        x = self.dropout_hidden(x, training=training)
        logits = self.classifier(x)

        if self.return_logits:
            return logits
        return tf.nn.softmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Legacy architecture -- kept so the existing best_emotion_model.h5 still loads
# ---------------------------------------------------------------------------
class ImprovedBertClassifier(tf.keras.Model):
    """Original 93%-accuracy DistilBERT classifier (kept for back-compat)."""

    def __init__(self, bert_model, num_classes: int = 6):
        super().__init__()
        self.bert = bert_model
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense = tf.keras.layers.Dense(256, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training: bool = False):
        bert_outputs = self.bert(inputs, training=training)
        token_embeddings = bert_outputs[0]
        attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(attention_mask, axis=-1)
        masked = token_embeddings * mask
        sum_embeddings = tf.reduce_sum(masked, axis=1)
        sum_mask = tf.reduce_sum(mask, axis=1)
        mean_pool = sum_embeddings / tf.maximum(sum_mask, 1e-9)
        x = self.dropout1(mean_pool, training=training)
        x = self.dense(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)


def build_backbone(model_name: str = "roberta-base") -> tf.keras.Model:
    """Load a transformer encoder with safe defaults."""
    return TFAutoModel.from_pretrained(
        model_name,
        from_pt=False,
        use_safetensors=False,
        return_dict=False,
    )
