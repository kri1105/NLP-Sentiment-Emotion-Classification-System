"""Minimal Streamlit UI — input text, get predicted emotion.

Preprocessing MUST mirror the notebook exactly, otherwise predictions are
garbage (model was trained on stopword-removed, lemmatized text).
"""

import json
import os
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# NLTK assets (already downloaded on first run; no network calls here)
from nltk.corpus import stopwords as _stopwords
from nltk.stem import WordNetLemmatizer


MODEL_PATH = "model/lstm_emotion/model.h5"
TOKENIZER_PATH = "model/lstm_emotion/tokenizer.json"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NONALPHA_RE = re.compile(r"[^a-z\s]")


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH) as f:
        cfg = json.load(f)
    tokenizer = tokenizer_from_json(cfg["tokenizer_json"])

    keep_words = set(cfg.get("keep_words",
                              ["not", "no", "never", "nor", "very", "n't"]))
    sw = set(_stopwords.words("english")) - keep_words
    lem = WordNetLemmatizer()
    lem.lemmatize("running")   # warm WordNet now, not at first predict

    return {
        "model": model,
        "tokenizer": tokenizer,
        "max_len": int(cfg.get("max_len", 40)),
        "labels": cfg.get(
            "emotion_labels",
            ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        ),
        "stopwords": sw,
        "lemmatizer": lem,
    }


def preprocess(text: str, sw: set, lem) -> str:
    """Exactly mirrors the training-time preprocessing."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NONALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in sw and len(t) > 1]
    tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def predict(text: str, a) -> tuple[str, float, str]:
    clean = preprocess(text, a["stopwords"], a["lemmatizer"])
    seq = a["tokenizer"].texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=a["max_len"],
                           padding="post", truncating="post")
    probs = a["model"].predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    return a["labels"][idx], float(probs[idx]), clean


EMOJI = {
    "Sadness":  "😢", "Joy": "😊", "Love": "❤️",
    "Anger":    "😠", "Fear": "😨", "Surprise": "😲",
}
COLOR = {
    "Sadness":  "#4A90E2", "Joy": "#F5A623", "Love": "#E94B8B",
    "Anger":    "#D0021B", "Fear": "#7B68EE", "Surprise": "#50E3C2",
}


# ─── UI ─────────────────────────────────────────────────────────────────────
st.title("Emotion Classifier")

if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
    st.error(f"Missing files:\n- {MODEL_PATH}\n- {TOKENIZER_PATH}")
    st.stop()

assets = load_assets()

text = st.text_area("Enter text:", height=120)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        label, conf, clean = predict(text, assets)
        emoji = EMOJI.get(label, "🎯")
        color = COLOR.get(label, "#1E88E5")

        st.markdown(
            f"""
<div style="
    margin-top: 1.25rem;
    padding: 1.5rem 1.75rem;
    border-radius: 12px;
    border-left: 6px solid {color};
    background: rgba(255,255,255,0.03);
">
    <div style="font-size: 0.85rem; color: #888;
                text-transform: uppercase; letter-spacing: 0.05em;
                margin-bottom: 0.5rem;">Predicted emotion</div>
    <div style="display: flex; align-items: baseline; gap: 0.75rem;">
        <span style="font-size: 2.6rem;">{emoji}</span>
        <span style="font-size: 2.2rem; font-weight: 700; color: {color};">
            {label}
        </span>
    </div>
    <div style="margin-top: 0.5rem; font-size: 1.05rem; color: #aaa;">
        Confidence:
        <span style="color: {color}; font-weight: 600;">{conf:.1%}</span>
    </div>
    <div style="margin-top: 0.75rem; height: 6px; background: #1a1f2e;
                border-radius: 3px; overflow: hidden;">
        <div style="height: 100%; width: {conf*100:.1f}%;
                    background: {color};"></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"After preprocessing: `{clean or '(empty)'}`")
