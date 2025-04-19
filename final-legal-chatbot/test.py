import os
import re
import fitz  
import spacy
import streamlit as st
from gpt4all import GPT4All  


llm = GPT4All("mistral-7b.Q4_0.gguf")  

# SpaCy NLP Model
nlp = spacy.load("en_core_web_sm")


def extract_text(file):
    """Extracts text from PDF or TXT legal documents."""
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    elif file.name.endswith(".txt"):
        text = file.getvalue().decode("utf-8")
    return text.strip()[:5000]  # Limit text to 5000 characters

def extract_legal_entities(text):
    """Extracts legal-related entities using SpaCy."""
    doc = nlp(text)
    entities = {"ORG": [], "PERSON": [], "DATE": [], "MONEY": [], "LAW": [], "GPE": []}
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return {key: list(set(value)) for key, value in entities.items()}  # Remove duplicates

# Legal Risk Using GPT4All
def analyze_legal_risks(text, entities):
    """Uses GPT4All (Mistral) to analyze risks and assign a risk score."""
    prompt = f"""
    You are a legal expert AI. Analyze this legal document for risks.
    
    - Document: {text[:3000]}...
    - Extracted Legal Entities: {entities}

    1. Identify missing or problematic clauses.
    2. Check for ambiguous terms that increase legal risk.
    3. Predict a **Legal Risk Score (1-10)** and provide recommendations.
    """

    response_text = llm.generate(prompt)
    risk_score = extract_risk_score(response_text)
    return risk_score, response_text

def extract_risk_score(response):
    match = re.search(r"\b([1-9]|10)\b", response)  # Extract numbers 1-10
    return int(match.group(1)) if match else 5  # Default risk score = 5


def main():
    st.title("📜 Legal Document Risk Analyzer")
    st.write("Upload a legal document (PDF or TXT) to analyze its risk.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        st.subheader("📂 Extracting text...")
        extracted_text = extract_text(uploaded_file)

        st.subheader("🔍 Extracting legal entities...")
        legal_entities = extract_legal_entities(extracted_text)
        st.json(legal_entities)

        st.subheader("⚖️ Analyzing legal risks...")
        risk_score, risk_analysis = analyze_legal_risks(extracted_text, legal_entities)

        st.subheader("⚠️ Legal Risk Score: ")
        st.markdown(f"## {risk_score}/10")

        st.subheader("📌 Risk Analysis Report:")
        st.write(risk_analysis)

if __name__ == "__main__":
    main()
