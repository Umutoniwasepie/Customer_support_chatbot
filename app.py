import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

# Load the model and tokenizer from Hugging Face Hub
MODEL_NAME = "Umutoniwasepie/final_model/tree/main/final_model"

@st.cache_resource  # Cache the model to avoid reloading on every run
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def normalize_input(text):
    """Cleans and normalizes user input."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

def capitalize_response(response):
    """Capitalizes and removes duplicate sentences from the response."""
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

def test_query(query):
    """Generates a response using the model."""
    query_lower = normalize_input(query)
    input_text = f"generate response: Current query: {query_lower}"
    
    input_ids = tokenizer(
        input_text, return_tensors="pt", truncation=True, 
        padding="max_length", max_length=128
    ).input_ids

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return capitalize_response(response)

# Streamlit UI
st.title("Customer Support Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send"):
    response = test_query(user_input)
    st.write(f"**Bot:** {response}")
