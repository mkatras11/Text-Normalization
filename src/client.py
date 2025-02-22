import streamlit as st
import requests

# FastAPI endpoints
LLM_ENDPOINT = "http://localhost:8000/normalize_llm"
NER_ENDPOINT = "http://localhost:8000/normalize_ner"

def normalize_text_llm(text):
    response = requests.post(LLM_ENDPOINT, json={"text": text})
    if response.status_code == 200:
        return response.json().get("normalized_text", "")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return ""

def normalize_text_ner(text):
    response = requests.post(NER_ENDPOINT, json={"text": text})
    if response.status_code == 200:
        return response.json().get("normalized_text", "")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return ""

# Streamlit UI
st.title("Text Normalization")

model_choice = st.radio(
    "Choose the normalization model:",
    ("LLM-based Normalization", "NER-based Normalization")
)

text_input = st.text_area("Enter text to normalize:")

if st.button("Normalize Text"):
    if text_input.strip() == "":
        st.warning("Please enter some text to normalize.")
    else:
        with st.spinner("Normalizing..."):
            if model_choice == "LLM-based Normalization":
                normalized_text = normalize_text_llm(text_input)
            else:
                normalized_text = normalize_text_ner(text_input)
            
            st.success("Normalization complete!")
            st.text_area("Normalized Text:", value=normalized_text, height=200)
