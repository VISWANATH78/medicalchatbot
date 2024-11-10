# Multilingual Healthcare Chatbot with Generative AI

## Overview
This project provides a multilingual healthcare chatbot leveraging speech recognition, language translation, and generative AI models for responding to user queries. The chatbot supports both typed and voice inputs, as well as audio and video file uploads (MP3/MP4). It uses a combination of GPT-2 for generating responses and Google's translation services to handle multiple languages.

## Features
- **Multilingual Support:** Automatically detects the language of the user’s input and translates it to English, processes it, and translates the response back to the original language.
- **Voice Input:** Captures and converts spoken queries into text using speech recognition.
- **File Upload Support:** Users can upload MP3 and MP4 files, from which speech is extracted and converted into text.
- **Generative AI Responses:** Utilizes GPT-2 to generate meaningful responses based on the user’s input.

## Workflow

### Step 1: **Input Method**
Users can choose one of the following input methods:
- **Text Input:** Directly type their query.
- **Voice Input:** Click the "Start Listening" button to speak the query.
- **File Upload:** Upload an audio or video file (MP3/MP4).

### Step 2: **Language Detection and Translation**
Once the input is received, the following steps are performed:
- **Language Detection:** The input text is analyzed to determine its language using `langdetect`.
- **Translation to English:** If the input is not in English, it is automatically translated into English using Google Translate's API (`googletrans`).

### Step 3: **Response Generation**
The chatbot generates a response to the user's query:
- **Text Generation:** The translated query is passed through the GPT-2 model (`GPT2LMHeadModel` from Hugging Face Transformers).
- **Response in English:** The response is first generated in English.

### Step 4: **Translate Response Back**
- **Back Translation:** The generated response is then translated back into the original language using Google Translate if necessary.

### Step 5: **Output**
The final generated response is shown to the user in the language they initially used.

---

## Technologies Used

1. **Streamlit** – For building the user interface.
2. **Google T5 Translator** – For language translation.
3. **GPT-2** – For text generation (using the Hugging Face `transformers` library).
4. **Speech Recognition** – To capture and transcribe speech input.
5. **PySoundDevice** – To record audio using sound devices.
6. **MoviePy** – To extract audio from video files (MP4).
7. **Pydub** – For audio format conversion (MP3 to WAV).
8. **Langdetect** – For language detection in the input text.

---

## Installation

To run this chatbot locally, follow the steps below.

### Prerequisites

- Python 3.x
- Install required libraries using pip:
  
```bash
pip install streamlit transformers googletrans==4.0.0-rc1 langdetect speechrecognition sounddevice pydub moviepy
```

### Steps to Run

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and visit `http://localhost:8501` to interact with the chatbot.

---

## How to Use

1. **Input Methods:** 
   - Choose whether you want to type a query, speak it, or upload an audio/video file.
   - If uploading a file, ensure it’s in MP3 or MP4 format.
   
2. **Generate Response:** 
   - After inputting the query, click on "Generate Response" to get the chatbot’s response.

3. **View Results:** 
   - The chatbot will display its response in your language, and you will see the original query as well as the translated and generated response.

---

## Example Workflow

1. **User Input:** "¿Cuál es el tratamiento para la fiebre?"
   - **Step 1:** The system detects the language as Spanish.
   - **Step 2:** Translates the question into English: "What is the treatment for fever?"
   - **Step 3:** GPT-2 generates a response in English: "The treatment for fever usually includes taking antipyretics like paracetamol, staying hydrated, and resting."
   - **Step 4:** Translates the response back to Spanish: "El tratamiento para la fiebre generalmente incluye tomar antipiréticos como paracetamol, mantenerse hidratado y descansar."
   - **Step 5:** The response is shown to the user in Spanish.

---

## Notes

- The voice recognition works best with clear audio and minimal background noise.
- The chatbot may take a few moments to process longer inputs or generate more complex responses.
- The uploaded files must be either MP3 or MP4 format for proper audio extraction.

---

## Contact

For any questions or feedback, please contact me at:  
**Name:** Viswanath V S  
**Email:** vichu110602@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive explanation of your chatbot application and how to set it up. It includes details about the workflow and the libraries used, making it easy for someone else to understand and replicate your work.
