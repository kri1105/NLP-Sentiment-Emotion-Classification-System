from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from transformers import AutoTokenizer, TFDistilBertModel

app = Flask(__name__)

# Architecture matches your notebook exactly
class BertClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_classes=6):
        super().__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        # We pass the input dict directly to the bert layer
        bert_outputs = self.bert(inputs, training=training)
        token_embeddings = bert_outputs[0]
        
        # Mean Pooling logic from your notebook
        attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(attention_mask, axis=-1)
        masked_embeddings = token_embeddings * mask
        mean_pool = tf.reduce_sum(masked_embeddings, axis=1) / tf.reduce_sum(mask, axis=1)
        
        x = self.dropout(mean_pool, training=training)
        return self.classifier(x)

# Load resources using the specific TF class
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Explicitly using TFDistilBertModel to fix Python 3.13 detection issues
bert_base = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
model = BertClassifier(bert_base)

# Build shapes before loading .h5 weights
model(tokenizer("initialization", return_tensors="tf"))

weights_path = "model/distilbert_emotion_classifier.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Model loaded successfully!")
else:
    print(f"File not found at {weights_path}")

labels = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

@app.route('/')
def index():
    return "Emotion API is Running. Post to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    preds = model(inputs, training=False).numpy()
    idx = np.argmax(preds)
    return jsonify({"emotion": labels[idx], "score": float(np.max(preds))})

if __name__ == '__main__':
    app.run(port=5000)