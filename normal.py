# import streamlit as st
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from langdetect import detect
# from deep_translator import GoogleTranslator

# # Load the Google Translator for translation (to detect and convert language to English)
# translator = GoogleTranslator()

# # Load GPT-2 model for generating responses
# gpt2_model_path = "model"  # Modify if you're using a custom model
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
# model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# # Function to detect language and translate to English
# def translate_to_english(text):
#     detected_lang = detect(text)
#     if detected_lang != 'en':
#         st.write(f"Detected Language: {detected_lang}")
#         st.write("Translating to English...")
#         translated_text = translator.translate(text, src=detected_lang, dest='en')
#         st.write(f"Translated to English: {translated_text}")
#         return translated_text
#     else:
#         return text

# # Function to generate response using GPT-2 model
# def generate_gpt2_response(prompt, model, tokenizer):
#     # Encode the input prompt
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
    
#     # Generate output using the model with controlled output length and randomness
#     outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
#     # Decode and return the generated text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

# # Function to check if the response is meaningful
# def is_meaningful_response(generated_response, original_input):
#     # If the response is too similar to the input, it is considered repetitive
#     if generated_response.strip().lower() == original_input.strip().lower():
#         return False
#     return True

# # Function to translate back to the detected language
# # Function to translate back to the detected language (adjusted for better handling of longer text)
# # Function to translate back to the detected language (adjusted for better handling of longer text)
# def translate_back_to_original(text, detected_lang, original_text):
#     if detected_lang != 'en':  # Only translate back if the detected language is not English
#         st.write(f"Translating back to {detected_lang}...")
        
#         # Translate in smaller chunks if the text is too long
#         max_chunk_length = 500  # Set a reasonable max length for each translation request
#         translated_text = ""
        
#         # Split the text into chunks to prevent long text errors
#         for i in range(0, len(text), max_chunk_length):
#             chunk = text[i:i+max_chunk_length]
#             translated_chunk = translator.translate(chunk, src='en', dest=detected_lang)
#             translated_text += translated_chunk  # Concatenate the translated chunks
        
#         # Handle case where translated text is similar to the original input (in case of a short or repetitive response)
#         if translated_text == original_text:
#             st.write(f"The generated response is too short or repetitive, trying to rephrase...")
#             translated_text = translator.translate("Sorry, I couldn't generate a detailed answer. Could you please rephrase your question?", src='en', dest=detected_lang)

#         st.write(f"Final Generated Response (translated back to {detected_lang}): {translated_text}")
#         return translated_text
#     else:
#         return text  # Return the English text if the language is already English

# # Streamlit UI for the chatbot
# def main():
#     st.title("Multilingual Healthcare Chatbot with Generative AI")
    
#     # Text input for the user
#     user_input = st.text_area("Ask your question:")
    
#     if st.button("Generate Response"):
#         if user_input:
#             with st.spinner("Processing..."):
#                 # Step 1: Detect language and translate to English
#                 translated_input = translate_to_english(user_input)

#                 # Step 2: Generate a more focused response using GPT-2
#                 # Update the prompt to be more specific: asking only about the symptoms of malaria
#                 prompt = "List the typical symptoms of malaria, including fever, chills, fatigue, and other common signs of infection."
#                 generated_response = generate_gpt2_response(prompt, model_gpt2, tokenizer_gpt2)
#                 st.write(f"Generated Response (in English): {generated_response}")

#                 # Step 3: Ensure the response is meaningful
#                 if not is_meaningful_response(generated_response, user_input):
#                     st.write("The response is too similar to the question. Trying to generate a more relevant response...")
#                     # Regenerate response with more focused detail
#                     prompt = "Explain the key symptoms of malaria with a focus on common signs like fever, chills, and fatigue."
#                     generated_response = generate_gpt2_response(prompt, model_gpt2, tokenizer_gpt2)
#                     st.write(f"New Generated Response: {generated_response}")

#                 # Step 4: Translate back to the original language
#                 detected_lang = detect(user_input)
#                 final_response = translate_back_to_original(generated_response, detected_lang, user_input)
                
#                 st.write(f"Final Response (in {detected_lang}): {final_response}")
#         else:
#             st.warning("Please enter a question.")

#     # Footer with your name and email
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("""<p style="text-align: center;">All rights reserved. <br>Viswanath V S | vichu110602@gmail.com</p>""", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Load MarianMT model for translation
def load_translation_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to translate text using the MarianMT model
def translate_text_nlp(text, src_lang, tgt_lang):
    model, tokenizer = load_translation_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Load GPT-2 model for generating responses
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate response using GPT-2
def generate_gpt2_response(prompt):
    inputs = tokenizer_gpt2.encode(prompt, return_tensors="pt")
    outputs = model_gpt2.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    generated_text = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit Interface
def main():
    st.title("Multilingual Chatbot")

    # Text input from the user
    user_input = st.text_area("Ask your question:")

    if st.button("Generate Response"):
        if user_input:
            # Step 1: Detect the language of the user's input
            detected_lang = detect(user_input)
            st.write(f"Detected Language: {detected_lang}")

            # Step 2: Translate the input to English if it's not already in English
            if detected_lang != 'en':
                translated_input = translate_text_nlp(user_input, detected_lang, 'en')
                st.write(f"Translated to English: {translated_input}")
            else:
                translated_input = user_input

            # Step 3: Generate the response using GPT-2
            generated_response = generate_gpt2_response(translated_input)
            st.write(f"Generated Response (in English): {generated_response}")

            # Step 4: Translate the response back to the original language
            if detected_lang != 'en':
                final_response = translate_text_nlp(generated_response, 'en', detected_lang)
                st.write(f"Final Response (in {detected_lang}): {final_response}")
            else:
                final_response = generated_response

        else:
            st.warning("Please enter a question.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<p style="text-align: center;">All rights reserved. <br>Viswanath V S | vichu110602@gmail.com</p>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
