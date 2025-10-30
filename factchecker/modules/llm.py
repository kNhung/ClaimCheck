import os
import ollama
import base64

def prompt_ollama(prompt, model='qwen2.5:0.5b', think=True):
    messages = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({
        'role': 'user',
         'content': prompt,
    })

    response = ollama.chat(model=model, messages=messages)

    output = response['message']['content']

    if '<think>' in output[:20]:
        output = output.split('</think>')[-1]
        
    return output

