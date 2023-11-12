npm install --save dotenv 

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import os
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

# image2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    You are a story teller;
    you can generate a short story based on a simple narrative, the story should be no more than 28 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables={"scenario"})

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

# text2speech
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/speechbrain/tts-tacotron2-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    
    payloads = {
        "inputs": message,
        "options": {
            "wait_for_model": True  # Add the wait_for_model option
        }
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    
    if response.status_code == 200:
        audio_data = response.content
        with open('audio.mp3', 'wb') as file:
            file.write(audio_data)
        print("Audio file saved as 'audio.mp3'")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Call the function with your message
scenario = img2text("cat.jpeg")
story = generate_story(scenario)
text2speech(story)

def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="kk")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("choose an image...", type="jpeg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.mp3")

if __name__ == '__main__':
    main()

