import openai
import os
import ollama
import base64
import google.generativeai as genai

def _configure_gemini(key_number=1):
    api_key = os.getenv(f"GEMINI_API_KEY_{key_number}")
    if not api_key:
        raise EnvironmentError(f"Set GEMINI_API_KEY_{key_number} before invoking Gemini.")
    genai.configure(api_key=api_key)

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

def prompt_gemini(prompt_text, model_name, think=True, key_number=1):
    _configure_gemini(key_number=key_number)
    system_instruction = None
    if not think:
        # Gemini does not support /no_think, so we enforce it via a system hint.
        system_instruction = (
            "Respond with the final answer only; do not include intermediate reasoning "
            "or <think> annotations."
        )

    generative_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    response = generative_model.generate_content(prompt_text)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()

def prompt_ollama(prompt, model_name, think=True):
#def prompt_ollama(prompt, model='qwen3:0.6b', think=True):
#def prompt_ollama(prompt, model='phi3:3.8b-mini-128k-instruct-q2_K', think=True):
#def prompt_ollama(prompt, model='jdevasier/qwen2.5-fact-verification', think=True):
#def prompt_ollama(prompt, model='qwen3:4b', think=True):
#def prompt_ollama(prompt, model='qwen3:1.7b', think=True):
#def prompt_ollama(prompt, model='qwen2.5:1.5b', think=True):
#def prompt_ollama(prompt, model='gpt-oss:20b-cloud', think=True):
#def prompt_ollama(prompt, model='deepseek-r1:1.5b', think=True):
#def prompt_ollama(prompt, model='qwen2.5:0.5b-instruct', think=True):
#def prompt_ollama(prompt, model='qwen2.5:0.5b-instruct-q4_0', think=True):

    # client = ollama.Client(
    #     host="https://ollama.com",
    #     headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    # )
    
    messages = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({
        'role': 'user',
        'content': prompt,
    })

    response = ollama.chat(model=model_name, messages=messages)
    #response = client.chat(model=model, messages=messages)

    output = response['message']['content']

    if '<think>' in output[:20]:
        output = output.split('</think>')[-1]
        
    return output


def prompt_model(prompt, model_name, think=True, key_number=1):
    if model_name.startswith('gemini'):
        return prompt_gemini(prompt, model_name=model_name, think=think, key_number=key_number)
    elif model_name.startswith('qwen') or model_name.startswith('phi') or model_name.startswith('gpt-oss') or model_name.startswith('deepseek'):
        return prompt_ollama(prompt, model_name=model_name, think=think)
    else:
        return prompt_gpt(prompt, model=model_name)