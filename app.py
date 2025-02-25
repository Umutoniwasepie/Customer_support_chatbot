# Import the necessary functions
import streamlit as st
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    import re
    import os
    import gdown
except ImportError as e:
    st.error(f"Error importing libraries: {e}. Please ensure 'transformers==4.36.0', 'torch==2.1.2', and 'gdown>=4.6.0' are installed in requirements.txt.")
    st.stop()

# Function to download a folder from Google Drive as a zip and extract it
def download_folder_from_drive(folder_id, output_dir):
    """Download a Google Drive folder and extract its contents directly."""
    # Generate a URL for the folder (gdown treats folders as zips but extracts to a directory)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        # gdown.download_folder() creates a directory with the folder's contents
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {e}")
        st.stop()
    
    # Extract the zip to the desired directory
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(output_dir))
    os.remove(zip_path)  # Clean up the zip file

# Google Drive folder ID (from link: https://drive.google.com/drive/folders/1V77zA-4JkDFUXZJcfIK1dDtdIDbPRb8H)
DRIVE_FOLDER_ID = "1V77zA-4JkDFUXZJcfIK1dDtdIDbPRb8H"

# Download and extract final_model and tokenized_dataset if not already present
MODEL_PATH = "final_model"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading final_model folder from Google Drive...")
    download_folder_from_drive(DRIVE_FOLDER_ID, MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory '{MODEL_PATH}' not downloaded or extracted correctly.")
        st.stop()

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

def capitalize_response(response):
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

def test_query(query):
    query_lower = normalize_input(query)
    input_text = f"generate response: Current query: {query_lower}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids
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

# Streamlit app layout
st.title("Customer Support Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send"):
    response = test_query(user_input)
    st.write(f"**Bot:** {response}")
