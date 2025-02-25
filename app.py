# Import necessary libraries
import streamlit as st
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    import re
    import os
    import gdown
except ImportError as e:
    st.error(f"Error importing libraries: {e}. Ensure 'transformers==4.36.0', 'torch==2.2.2', and 'gdown>=4.6.0' are installed in requirements.txt.")
    st.stop()

# Function to download a folder from Google Drive
def download_folder_from_drive(folder_id, output_dir):
    """Download a Google Drive folder and extract its contents directly."""
    try:
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {e}")
        st.stop()

# Google Drive folder ID
DRIVE_FOLDER_ID = "1V77zA-4JkDFUXZJcfIK1dDtdIDbPRb8H"

# Define model and dataset paths
MODEL_PATH = "final_model"
DATASET_PATH = os.path.join(MODEL_PATH, "tokenized_dataset")  # Assuming dataset is inside final_model

# Download and extract final_model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    download_folder_from_drive(DRIVE_FOLDER_ID, MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory '{MODEL_PATH}' not downloaded correctly.")
        st.stop()

# Load the model and tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Move model to CPU (Streamlit Cloud defaults to CPU)
device = torch.device("cpu")
model.to(device)
model.eval()

# Helper Functions
def normalize_input(text):
    """Normalize text by lowercasing, removing special characters, and standardizing spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

def capitalize_response(response):
    """Capitalize the first letter of sentences for readability."""
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

def test_query(query):
    """Generate a response for a customer query using T5-small."""
    query_lower = normalize_input(query)
    input_text = f"customer support query: {query_lower}"
    
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids.to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5,
            num_return_sequences=1,  # Prevents warnings
            do_sample=True
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return capitalize_response(response)

# Streamlit UI
st.title("Customer Support Chatbot")
st.write("Ask any customer support-related question, such as order issues, refunds, or account management.")

user_input = st.text_input("You:", "")
if st.button("Send"):
    response = test_query(user_input)
    st.write(f"**Bot:** {response}")
