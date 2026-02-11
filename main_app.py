import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, TFDistilBertModel


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------
# Improved Model Architecture
# -------------------------
class ImprovedBertClassifier(tf.keras.Model):
    """
    Enhanced DistilBERT classifier with double dropout and additional dense layer.
    Achieves 93% accuracy on emotion classification.
    """
    def __init__(self, bert_model, num_classes=6):
        super().__init__()
        self.bert = bert_model
        # Double dropout for better regularization
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        # Additional dense layer for enhanced feature extraction
        self.dense = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        # Pass through BERT
        bert_outputs = self.bert(inputs, training=training)
        token_embeddings = bert_outputs[0]

        # Mean pooling with attention mask
        attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(attention_mask, axis=-1)
        masked_embeddings = token_embeddings * mask
        
        # Safe division to avoid NaN
        sum_embeddings = tf.reduce_sum(masked_embeddings, axis=1)
        sum_mask = tf.reduce_sum(mask, axis=1)
        mean_pool = sum_embeddings / tf.maximum(sum_mask, 1e-9)

        # Pass through classification head
        x = self.dropout1(mean_pool, training=training)
        x = self.dense(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

# -------------------------
# Load model + tokenizer
# -------------------------
@st.cache_resource
def load_model():
    """Load the improved emotion classification model."""

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_base = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    model = ImprovedBertClassifier(bert_base)

    # Build model
    dummy_input = tokenizer("init", return_tensors="tf", padding=True)
    model(dummy_input)

    improved_weights_path = "model/best_emotion_model.h5"

    if not os.path.exists(improved_weights_path):
        raise FileNotFoundError(
            "Model weights not found. Ensure model files are in the 'model/' directory."
        )

    model.load_weights(improved_weights_path)

    return tokenizer, model


with st.spinner("🔄 Loading model... This may take a moment on first run."):
    try:
        tokenizer, model = load_model()
        st.toast("✅ Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()



# -------------------------
# Initialize
# -------------------------
tokenizer, model = load_model()

# Emotion labels and emojis
emotion_info = {
    0: {"name": "Sadness", "emoji": "😢", "color": "#4A90E2"},
    1: {"name": "Joy", "emoji": "😊", "color": "#F5A623"},
    2: {"name": "Love", "emoji": "❤️", "color": "#E94B8B"},
    3: {"name": "Anger", "emoji": "😠", "color": "#D0021B"},
    4: {"name": "Fear", "emoji": "😨", "color": "#7B68EE"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#50E3C2"}
}


# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
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
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)



# -------------------------
# Main UI
# -------------------------
st.markdown('<h1 class="main-header">🧠 Emotion Detection with DistilBERT</h1>', unsafe_allow_html=True)

# Input section
st.markdown("### 💬 Enter Your Text")
text_input = st.text_area(
    "",
    placeholder="Type or paste any text here to analyze its emotional content...",
    height=150,
    value=st.session_state.get('example_text', ''),
    key='text_input'
)

# Clear example from session state after use
if 'example_text' in st.session_state:
    del st.session_state['example_text']

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    predict_button = st.button("🚀 Predict Emotion", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Prediction section
if predict_button and text_input.strip():
    with st.spinner("🔮 Analyzing emotion..."):
        # Tokenize
        inputs = tokenizer(
            text_input,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Predict
        predictions = model(inputs, training=False).numpy()[0]
        predicted_idx = np.argmax(predictions)
        predicted_emotion = emotion_info[predicted_idx]["name"]
        predicted_emoji = emotion_info[predicted_idx]["emoji"]
        confidence = predictions[predicted_idx]

    # Display results
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Main prediction box
    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
    
    st.markdown(f"""
    <div class="emotion-box">
        <h2 style="margin:0; color: {emotion_info[predicted_idx]['color']};">
            {predicted_emoji} {predicted_emotion}
        </h2>
        <p style="margin-top: 0.5rem; font-size: 1.2rem;">
            Confidence: <span class="{confidence_class}">{confidence:.2%}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed probability breakdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Probability Distribution")
        
        # Create dataframe for display
        prob_df = pd.DataFrame({
            'Emotion': [f"{emotion_info[i]['emoji']} {emotion_info[i]['name']}" for i in range(6)],
            'Probability': [f"{predictions[i]:.2%}" for i in range(6)],
            'Score': predictions
        }).sort_values('Score', ascending=False)
        
        # Display as table
        st.dataframe(
            prob_df[['Emotion', 'Probability']],
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 📈 Confidence Visualization")
        
        # Create interactive bar chart with Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=[predictions[i] for i in range(6)],
                y=[f"{emotion_info[i]['emoji']} {emotion_info[i]['name']}" for i in range(6)],
                orientation='h',
                marker=dict(
                    color=[emotion_info[i]['color'] for i in range(6)],
                    line=dict(color='white', width=2)
                ),
                text=[f"{predictions[i]:.1%}" for i in range(6)],
                textposition='outside',
            )
        ])
        
        fig.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Emotion",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>NLP Sentiment & Emotion Classification System</strong></p>
    <p>Built with DistilBERT • Trained on 16,000 samples • 93% Test Accuracy</p>
    <p>Created by Krithi S J | 
    <a href="https://github.com/kri1105" target="_blank">GitHub</a> | 
    <a href="mailto:krithi11505@gmail.com">Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)