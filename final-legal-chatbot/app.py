import streamlit as st
import fitz  
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import speech_recognition as sr


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

st.title("Legal Advisor 📚")


selected_feature = st.sidebar.radio("Select Feature", ("Chatbot", "Summarize PDF"))

if selected_feature == "Chatbot":
    
    selected_chatbot = st.sidebar.radio("Select Chatbot", ("OpenAI", "Llama 2"))
    if selected_chatbot == "OpenAI":
        from src import openai_call
    elif selected_chatbot == "Llama 2":
        st.warning(
            "It might take some time to get response because of the size of Llama 2 model ⚠️"
        )
        from src import llama_call


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def speech_to_text():
        recognizer = sr.Recognizer()
        mic = sr.Microphone(device_index=1)  
        with mic as source:
            st.info("Listening... 🎙️")
            recognizer.adjust_for_ambient_noise(source)

            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                st.success(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Sorry, could not understand the audio.")
            except sr.RequestError:
                st.error("Could not request results, check your internet connection.")
            except sr.WaitTimeoutError:
                st.error("No speech detected, please try again.")
        return ""

    # Voice input button
    if st.button("🎤 Speak"):
        spoken_text = speech_to_text()
        if spoken_text:
            st.session_state["spoken_text"] = spoken_text

    # User text input
    prompt = st.chat_input("Ask something about law") or st.session_state.pop("spoken_text", "")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking ✨..."):
            if selected_chatbot == "Llama 2":
                response = llama_call(prompt)
            elif selected_chatbot == "OpenAI":
                response = openai_call(prompt)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif selected_feature == "Summarize PDF":
    st.subheader("📄 Upload a PDF to Summarize")
    
    # PDF file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        def extract_text_from_pdf(file):
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as pdf_document:
                for page in pdf_document:
                    text += page.get_text("text")
            return text.strip()

        def summarize_text(text, max_length=150, min_length=30):
            if not text:
                return "No text extracted from the document."
            
            inputs = tokenizer.encode("summary: " + text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Extract text from the PDF
        with st.spinner("Extracting text... ⏳"):
            pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            with st.spinner("Summarizing... ⏳"):
                summary = summarize_text(pdf_text)

            st.subheader("📌 Summary:")
            st.write(summary)
