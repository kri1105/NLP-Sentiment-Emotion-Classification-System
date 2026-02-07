# app.py
import streamlit as st
import pickle
import numpy as np

# Load your model
@st.cache_resource
def load_model():
    with open('your_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# UI Components
st.title("Your NLP Model Interface")
st.write("Enter text for processing:")

# Input
user_input = st.text_area("Input Text:", height=150)

# Process button
if st.button("Process"):
    if user_input:
        with st.spinner("Processing..."):
            # Call your model
            prediction = model.predict([user_input])
            st.success("Done!")
            st.write("Results:", prediction)
    else:
        st.warning("Please enter some text")