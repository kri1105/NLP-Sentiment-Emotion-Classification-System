# Text-Based Emotion Detection System

**DistilBERT · TensorFlow · HuggingFace**

## Overview

This project is a **Text-Based Emotion Detection System** that classifies user text into one of six human emotions using a fine-tuned **DistilBERT** transformer model.

The system leverages **HuggingFace Transformers** and **TensorFlow (Keras)** to build an end-to-end NLP pipeline, including tokenization, model training, evaluation, and inference. Special attention is given to **class imbalance**, ensuring that minority emotions such as *surprise* and *love* are not ignored by the model.

The model is trained on the **DAIR-AI Emotion Dataset** and evaluated using accuracy, confusion matrix, and detailed classification metrics.

## 🎭 Emotion Classes

| Label ID | Emotion  |
|----------|----------|
| 0        | Sadness  |
| 1        | Joy      |
| 2        | Love     |
| 3        | Anger    |
| 4        | Fear     |
| 5        | Surprise |

## ✨ Key Features

- **Transformer-based Architecture**: Uses DistilBERT, a lightweight and efficient version of BERT, for contextual text understanding
- **Fine-Tuning Enabled**: DistilBERT layers are unfrozen to allow task-specific learning for emotion classification
- **Mean Pooling Strategy**: Since DistilBERT does not provide a `pooler_output`, mean pooling over token embeddings is implemented using attention masks
- **Class Imbalance Handling**: Custom class weights are applied during training to improve recall for underrepresented emotions
- **Efficient Data Pipeline**: Built using `tf.data.Dataset` with shuffling, batching, and prefetching for optimal performance
- **Early Stopping Regularization**: Prevents overfitting by restoring the best validation model automatically
- **Comprehensive Evaluation**: Includes confusion matrix, precision, recall, F1-score, and test set evaluation
- **Inference Support**: A reusable function allows emotion prediction on any custom text input

## 🛠 Tech Stack

| Component              | Tool / Library                    |
|------------------------|-----------------------------------|
| **Language**           | Python 3                          |
| **Deep Learning**      | TensorFlow (Keras)                |
| **Transformer Models** | HuggingFace Transformers          |
| **Pretrained Model**   | DistilBERT (`distilbert-base-uncased`) |
| **Dataset**            | DAIR-AI Emotion Dataset           |
| **Tokenization**       | HuggingFace AutoTokenizer         |
| **Evaluation**         | Scikit-learn                      |
| **Visualization**      | Matplotlib                        |

## 📋 Prerequisites

Ensure the following are installed:

- Python 3.8+
- pip
- Git
- TensorFlow
- Transformers
- Datasets
- Scikit-learn

## 📁 Project Structure

text-emotion-detector/
│
├── app.py                     # Streamlit UI (Frontend)
│
├── models/
│   └── distilbert_emotion_classifier.h5   # Trained model
│
├── src/
│   ├── __init__.py
│   ├── tokenizer.py           # Tokenizer loading & preprocessing
│   ├── model_loader.py        # Model loading logic
│   ├── predict.py             # Emotion prediction function
│
├── notebooks/
│   └── training.ipynb         # (Optional) Training notebook
│
├── requirements.txt
├── README.md
└── .gitignore



# HOW TO USE
1. Open a terminal
2. `cd <folder-name-of-this-project>`
3.  `python -m venv venv`
4. `.\venv\scripts\activate`
5. `pip install -r requirements.txt`
6. `streamlit run main_app.py`
7. Wait a little bit for everything to show up
