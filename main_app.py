"""Streamlit UI for the NLP Sentiment & Emotion Classification System.

The heavy lifting (model architecture, loading, prediction) lives in `src/`.
This file is now just the presentation layer. It automatically picks up the
improved RoBERTa weights once they're saved to
`model/best_emotion_model_roberta.h5`, and falls back to the legacy DistilBERT
weights otherwise — no code changes needed when you retrain.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `from src.* import ...` regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_loader import load_emotion_model
from src.predict import EMOTION_INFO, predict_emotion


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Cached model loading
# -------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model():
    return load_emotion_model()


with st.spinner("🔄 Loading model... This may take a moment on first run."):
    try:
        loaded = load_cached_model()
        st.toast(f"✅ Model loaded ({loaded.backbone_name})")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


# -------------------------
# Custom CSS
# -------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }
    .confidence-high   { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low    { color: #F44336; font-weight: bold; }
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# Main UI
# -------------------------
st.markdown(
    '<h1 class="main-header">🧠 Emotion Detection</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="sub-header">Backbone: <code>{loaded.backbone_name}</code></p>',
    unsafe_allow_html=True,
)

st.markdown("### 💬 Enter Your Text")
text_input = st.text_area(
    "",
    placeholder="Type or paste any text here to analyze its emotional content...",
    height=150,
    value=st.session_state.get("example_text", ""),
    key="text_input",
)
if "example_text" in st.session_state:
    del st.session_state["example_text"]

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    predict_button = st.button("🚀 Predict Emotion", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()


# -------------------------
# Prediction
# -------------------------
if predict_button and text_input.strip():
    with st.spinner("🔮 Analyzing emotion..."):
        predictions = predict_emotion(loaded, text_input)
        predicted_idx = int(np.argmax(predictions))
        predicted_emotion = EMOTION_INFO[predicted_idx]["name"]
        predicted_emoji = EMOTION_INFO[predicted_idx]["emoji"]
        confidence = float(predictions[predicted_idx])

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    confidence_class = (
        "confidence-high"
        if confidence > 0.8
        else "confidence-medium"
        if confidence > 0.6
        else "confidence-low"
    )
    st.markdown(
        f"""
    <div class="emotion-box">
        <h2 style="margin:0; color: {EMOTION_INFO[predicted_idx]['color']};">
            {predicted_emoji} {predicted_emotion}
        </h2>
        <p style="margin-top: 0.5rem; font-size: 1.2rem;">
            Confidence: <span class="{confidence_class}">{confidence:.2%}</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📊 Probability Distribution")
        prob_df = pd.DataFrame(
            {
                "Emotion": [
                    f"{EMOTION_INFO[i]['emoji']} {EMOTION_INFO[i]['name']}"
                    for i in range(6)
                ],
                "Probability": [f"{predictions[i]:.2%}" for i in range(6)],
                "Score": predictions,
            }
        ).sort_values("Score", ascending=False)
        st.dataframe(
            prob_df[["Emotion", "Probability"]],
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("### 📈 Confidence Visualization")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[predictions[i] for i in range(6)],
                    y=[
                        f"{EMOTION_INFO[i]['emoji']} {EMOTION_INFO[i]['name']}"
                        for i in range(6)
                    ],
                    orientation="h",
                    marker=dict(
                        color=[EMOTION_INFO[i]["color"] for i in range(6)],
                        line=dict(color="white", width=2),
                    ),
                    text=[f"{predictions[i]:.1%}" for i in range(6)],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Emotion",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    f"""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>NLP Sentiment & Emotion Classification System</strong></p>
    <p>Powered by <code>{loaded.backbone_name}</code> • Trained on dair-ai/emotion</p>
    <p>Created by Krithi S J |
    <a href="https://github.com/kri1105" target="_blank">GitHub</a> |
    <a href="mailto:krithi11505@gmail.com">Contact</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
