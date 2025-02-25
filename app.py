# Import necessary libraries
import streamlit as st
import os
import re
import torch
import gdown
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError as e:
    st.error(f"Error importing libraries: {e}. Ensure 'transformers==4.36.0', 'torch==2.2.2', and 'gdown>=4.6.0' are installed in requirements.txt.")
    st.stop()

# Ensure SentencePiece is installed (required for T5 tokenizer)
try:
    import sentencepiece
except ImportError:
    st.error("SentencePiece is required for T5Tokenizer but is missing. Install it using: pip install sentencepiece")
    st.stop()

# Google Drive folder IDs (replace these with your actual folder IDs)
FINAL_MODEL_ID = "10YOHPxJq80tP_pMmMOA__0BzRDLcZ5XG"
TOKENIZED_DATASET_ID = "1tuFVO3bYyFRW3PNmNvYlRvYr3o_zIfZ5"  

# Paths to save the downloaded data
MODEL_PATH = "final_model"
DATASET_PATH = "tokenized_dataset"

# Function to download a folder from Google Drive
def download_folder(folder_id, output_dir):
    """Download a folder from Google Drive and extract its contents."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"Error downloading '{output_dir}' from Google Drive: {e}")
        st.stop()

# Download final_model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading 'final_model' from Google Drive...")
    download_folder(FINAL_MODEL_ID, MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory '{MODEL_PATH}' not downloaded correctly.")
        st.stop()

# Download tokenized_dataset if not already present
if not os.path.exists(DATASET_PATH):
    st.info("Downloading 'tokenized_dataset' from Google Drive...")
    download_folder(TOKENIZED_DATASET_ID, DATASET_PATH)
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset directory '{DATASET_PATH}' not downloaded correctly.")
        st.stop()

# Load the tokenizer and model
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Move model to CPU
device = torch.device("cpu")
model.to(device)
model.eval()

# Utility functions
def normalize_input(text):
    """Normalize text: lowercasing, removing special characters, and standardizing spaces."""
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
    input_text = f"generate response: Current query: {query_lower}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids.to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5,
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
