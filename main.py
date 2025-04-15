import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json
import requests

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

gpt_model = "gpt-4o-mini"
ollama_model = "llama3.1"

system_message = "You are a helpful meteorologist bot"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather forecast for a given coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        }
    }
}]

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=gpt_model, messages=messages, tools=tools)
    
    if response.choices[0].finish_reason=="tool_calls": #checks if model needs to use tool
        message = response.choices[0].message
        response = handle_tool_call(message) #runs get_weather function
        messages.append(message)
        messages.append(response)
        stream = openai.chat.completions.create(model=gpt_model, messages=messages, stream=True)
        
    else:
        # Just stream the original response
        messages.append(response.choices[0].message)
        stream = openai.chat.completions.create(
            model=gpt_model,
            messages=messages,
            stream=True
        )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        yield result

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments) #gpt sends arguments as JSON string, not a dict, this line converts it so you can work with it
    result = get_weather(args["latitude"], args["longitude"])
    response = {
        "role": "tool",
        "content": json.dumps({"temperature": result}),
        "tool_call_id": tool_call.id
    }
    return response


gr.ChatInterface(fn=chat, type="messages").launch(inbrowser=True)














