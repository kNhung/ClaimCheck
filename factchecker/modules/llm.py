import openai
import os
import ollama
import base64
import google.generativeai as genai

def prompt_gpt(prompt, model='o4-mini-2025-04-16'):
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    client = openai.OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def prompt_gemini(prompt, model='gemini-2.5-flash'):
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

    response = genai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

#def prompt_ollama(prompt, model='qwen2.5:0.5b', think=True):
#def prompt_ollama(prompt, model='qwen3:0.6b', think=True):
#def prompt_ollama(prompt, model='jdevasier/qwen2.5-fact-verification', think=True):
#def prompt_ollama(prompt, model='qwen3:4b', think=True):
#def prompt_ollama(prompt, model='qwen3:1.7b', think=True):
#def prompt_ollama(prompt, model='gpt-oss:20b-cloud', think=True):
#def prompt_ollama(prompt, model='deepseek-r1:1.5b', think=True):
def prompt_ollama(prompt, model='qwen2.5:0.5b-instruct', think=True):

    client = ollama.Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )
    
    messages = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({
        'role': 'user',
         'content': prompt,
    })

    response = ollama.chat(model=model, messages=messages)
    #response = client.chat(model=model, messages=messages)

    output = response['message']['content']

    if '<think>' in output[:20]:
        output = output.split('</think>')[-1]
        
    return output

