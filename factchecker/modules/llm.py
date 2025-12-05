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
ACTION_NEEDED_MODEL = os.getenv("FACTCHECKER_ACTION_NEEDED_MODEL", None)  # Use faster model for action_needed if set
JUDGE_MODEL = os.getenv("FACTCHECKER_JUDGE_MODEL", None)  # Use faster model for judging if set
JUDGE_PROVIDER = os.getenv("FACTCHECKER_JUDGE_PROVIDER", "ollama").lower()  # "ollama" or "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Default Gemini model


def set_default_ollama_model(model_name: str):
    global DEFAULT_OLLAMA_MODEL
    if model_name:
        DEFAULT_OLLAMA_MODEL = model_name

def get_default_ollama_model():
    return DEFAULT_OLLAMA_MODEL


def prompt_ollama(prompt, model=None, think=True, use_action_needed_model=False, use_judge_model=False):
    if not model:
        if use_action_needed_model and ACTION_NEEDED_MODEL:
            model = ACTION_NEEDED_MODEL
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


def prompt_gemini(prompt, model=None, api_key=None):
    """
    Gọi Gemini API để xử lý prompt.
    
    Args:
        prompt: Prompt text
        model: Tên model Gemini (mặc định từ GEMINI_MODEL env var)
        api_key: API key (mặc định từ GEMINI_API_KEY env var)
    
    Returns:
        str: Response từ Gemini
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package is required for Gemini API. "
            "Install it with: pip install google-generativeai"
        )
    
    if not api_key:
        api_key = GEMINI_API_KEY
    
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it as an environment variable or in a .env file."
        )
    
    if not model:
        model = GEMINI_MODEL
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Create model instance
    gemini_model = genai.GenerativeModel(model)
    
    # Generate response
    response = gemini_model.generate_content(prompt)
    
    return response.text

