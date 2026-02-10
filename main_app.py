import streamlit as st
import tensorflow as tf
import numpy as np
import os
from transformers import AutoTokenizer, TFDistilBertModel


# -------------------------
# Model Architecture
# -------------------------
class BertClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_classes=6):
        super().__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        bert_outputs = self.bert(inputs, training=training)
        token_embeddings = bert_outputs[0]

        attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(attention_mask, axis=-1)
        masked_embeddings = token_embeddings * mask
        mean_pool = tf.reduce_sum(masked_embeddings, axis=1) / tf.reduce_sum(
            mask, axis=1
        )

        x = self.dropout(mean_pool, training=training)
        return self.classifier(x)


# -------------------------
# Load model + tokenizer
# -------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_base = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    model = BertClassifier(bert_base)

    # build model
    model(tokenizer("init", return_tensors="tf"))

    weights_path = "model/distilbert_emotion_classifier.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error(f"Model weights not found at {weights_path}")

    return tokenizer, model


tokenizer, model = load_model()

labels = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠")
st.title("🧠 Emotion Detection with DistilBERT")

text = st.text_area(
    "Enter text:", placeholder="Type something emotional here...", height=120
)

if st.button("Predict Emotion 🚀") and text.strip():
    inputs = tokenizer(
        text, return_tensors="tf", padding=True, truncation=True, max_length=128
    )

    preds = model(inputs, training=False).numpy()[0]
    idx = np.argmax(preds)

    st.success(f"**Emotion:** {labels[idx]}")
    st.write(f"**Confidence:** `{preds[idx]:.4f}`")

    st.bar_chart({labels[i]: float(preds[i]) for i in range(len(preds))})
