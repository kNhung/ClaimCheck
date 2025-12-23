import os
import ollama

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
JUDGE_PROVIDER = os.getenv("FACTCHECKER_JUDGE_PROVIDER", "ollama").lower()  # "ollama" or "gemini"
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", None)  # Use faster model for judging if set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Default Gemini model


def get_default_ollama_model():
    return DEFAULT_OLLAMA_MODEL


def prompt_ollama(prompt, model=None, think=True, use_judge_model=False):
    if use_judge_model and OLLAMA_JUDGE_MODEL:
        model = OLLAMA_JUDGE_MODEL
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
    
    # Track token usage từ Gemini API response
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, 'prompt_token_count', 0)
        completion_tokens = getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'completion_token_count', 0)
        total_tokens = getattr(usage, 'total_token_count', prompt_tokens + completion_tokens)
        
        print(f"[GEMINI TOKEN USAGE] Model: {model}")
        print(f"[GEMINI TOKEN USAGE] Prompt tokens: {prompt_tokens}")
        print(f"[GEMINI TOKEN USAGE] Completion tokens: {completion_tokens}")
        print(f"[GEMINI TOKEN USAGE] Total tokens: {total_tokens}")
    else:
        print(f"[GEMINI TOKEN USAGE] ⚠️  Token usage metadata not available in response")
    
    return response.text

