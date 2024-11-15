import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect
from googletrans import Translator
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import tempfile
import wavio
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
import os

# Load the Google T5 model for translation (to detect and convert language to English)
translator = Translator()

# Load GPT-2 model for generating responses
gpt2_model_path = "/Users/viswanathvs/Code/medical_chatbot/model/gpt2"
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to detect language and translate to English
def translate_to_english(text):
    detected_lang = detect(text)
    if detected_lang != 'en':
        st.write(f"Detected Language: {detected_lang}")
        st.write("Translating to English...")
        translated_text = translator.translate(text, src=detected_lang, dest='en').text
        st.write(f"Translated to English: {translated_text}")
        return translated_text
    else:
        return text

# Function to generate response using GPT-2 model
def generate_gpt2_response(prompt, model, tokenizer):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate output using the model
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to translate text back to the detected language
def translate_back_to_original(text, detected_lang):
    if detected_lang != 'en':
        st.write(f"Translating back to {detected_lang}...")
        translated_text = translator.translate(text, src='en', dest=detected_lang).text
        st.write(f"Final Generated Response: {translated_text}")
        return translated_text
    else:
        return text

# Function to capture voice input and convert it to text using pysounddevice
def voice_to_text():
    recognizer = sr.Recognizer()
    
    # Record audio using sounddevice
    st.write("Listening...")
    fs = 16000  # Sampling rate
    duration = 5  # seconds
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is done

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wavio.write(temp_file.name, fs, audio_data)
        audio_file = temp_file.name
    
    # Use speech_recognition to convert audio file to text
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service.")
        return ""

# Function to handle MP3 file uploads and convert to text
def mp3_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_mp3(audio_file)  # Convert MP3 to WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio.export(temp_file.name, format="wav")
        audio_file = temp_file.name

    # Use speech_recognition to convert audio file to text
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service.")
        return ""

# Function to handle MP4 file uploads and convert audio to text
def mp4_to_text(audio_file):
    recognizer = sr.Recognizer()
    
    # Extract audio from MP4 file
    audio_clip = AudioFileClip(audio_file)
    audio_clip.write_audiofile("temp_audio.wav", codec='pcm_s16le')

    # Use speech_recognition to convert audio file to text
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service.")
        return ""

# Function to convert MP3 to WAV format (if needed)
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_file.name, format="wav")
    return wav_file.name

# Function to convert MP4 to WAV format (if needed)
def convert_mp4_to_wav(mp4_file):
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_file.name, format="wav")
    return wav_file.name

# Function to perform speech recognition
def voice_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Error with speech recognition service.")
        return ""

# Streamlit UI for the chatbot
def main():
    st.title("Multilingual Healthcare Chatbot with Generative AI")
    
    # Option to either type, speak or upload an audio file
    input_method = st.radio("Choose input method:", ("Type", "Speak", "Upload Audio"))
    
    if input_method == "Speak":
        if st.button("Start Listening"):
            user_input = voice_to_text()
    elif input_method == "Upload Audio":
        uploaded_file = st.file_uploader("Choose an audio file (MP3/MP4)", type=["mp3", "mp4"])
        if uploaded_file is not None:
            if uploaded_file.type == "audio/mp3":
                user_input = mp3_to_text(uploaded_file)
            elif uploaded_file.type == "video/mp4":
                user_input = mp4_to_text(uploaded_file)
    else:
        user_input = st.text_area("Ask your question:")
    
    if st.button("Generate Response"):
        if user_input:
            with st.spinner("Processing..."):
                # Step 1: Detect language and translate to English
                translated_input = translate_to_english(user_input)

                # Step 2: Generate response using GPT-2
                generated_response = generate_gpt2_response(translated_input, model_gpt2, tokenizer_gpt2)
                st.write(f"Generated Response (in English): {generated_response}")

                # Step 3: Translate back to original language
                detected_lang = detect(user_input)
                final_response = translate_back_to_original(generated_response, detected_lang)
        else:
            st.warning("Please enter or speak a question.")

if __name__ == "__main__":
    main()

