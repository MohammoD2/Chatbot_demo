import streamlit as st
from safetensors import safe_open
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gdown
import os
# https://drive.google.com/file/d/1-oGXsn7E2FIivpWvMTm8ZazHoB9gzv-R/view?usp=sharing
# Define the Google Drive file ID for model.safetensors
GDRIVE_FILE_ID = "1-oGXsn7E2FIivpWvMTm8ZazHoB9gzv-R"


# Define the model path to save it locally
MODEL_PATH = "model.safetensors"

# Function to download the model from Google Drive if not already downloaded
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    download_model()
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # Use the base model config
    
    # Load weights from the safetensors file
    with safe_open(MODEL_PATH, framework="pt") as f:
        for name, param in model.named_parameters():
            param.data = f.get_tensor(name)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Function to generate a chatbot response
def generate_response(input_text, model, tokenizer):
    input_prompt = f"User: {input_text} <|endoftext|> Response:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in response:
        response = response.split("Response:")[1].strip()
    return response

# Streamlit UI
st.title("Chatbot")
st.write("Enter your prompt below to chat with the bot!")

# Load model and tokenizer
model, tokenizer = load_model()

# Get user input and generate response
user_input = st.text_input("Enter your prompt:")
if user_input:
    response = generate_response(user_input, model, tokenizer)
    st.write(f"Bot: {response}")
