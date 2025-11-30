import os
import ollama
import dotenv

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()

    print("Load env")
except Exception:
    pass

DEFAULT_OLLAMA_MODEL = os.getenv("FACTCHECK_MODEL_NAME", "qwen3:4b")
REASONING_MODEL = os.getenv("FACTCHECKER_REASONING_MODEL", None)  # Use faster model for reasoning if set
JUDGE_MODEL = os.getenv("FACTCHECKER_JUDGE_MODEL", None)  # Use faster model for judging if set


def set_default_ollama_model(model_name: str):
    global DEFAULT_OLLAMA_MODEL
    if model_name:
        DEFAULT_OLLAMA_MODEL = model_name

def get_default_ollama_model():
    return DEFAULT_OLLAMA_MODEL


def prompt_ollama(prompt, model=None, think=True, use_reasoning_model=False, use_judge_model=False):
    if not model:
        if use_reasoning_model and REASONING_MODEL:
            model = REASONING_MODEL
        elif use_judge_model and JUDGE_MODEL:
            model = JUDGE_MODEL
        else:
            model = DEFAULT_OLLAMA_MODEL
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

