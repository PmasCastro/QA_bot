import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import fitz  # PyMuPDF

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

MODEL = "gpt-4o-mini"

system_message = "You are a helpful Q&A bot."
system_message += "You specialize in Philosophy."
system_message += "You only answer questions related to philosophy"

# Load PDF text once at startup
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_text = extract_text_from_pdf("phildb.pdf")

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch(inbrowser=True)