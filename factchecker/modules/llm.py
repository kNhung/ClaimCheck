import os
import ollama
import base64
from groq import Groq
from .token_tracker import get_global_tracker, TokenUsage

# Try to import Groq exceptions
try:
    from groq import NotFoundError as GroqNotFoundError
except ImportError:
    try:
        from groq.errors import NotFoundError as GroqNotFoundError
    except ImportError:
        # Fallback: catch by error message
        GroqNotFoundError = None

DEFAULT_GROQ_MODEL = os.getenv("FACTCHECK_MODEL_NAME", "llama-3.1-8b-instant")
DEFAULT_OLLAMA_MODEL = os.getenv("FACTCHECK_MODEL_NAME", "qwen3:4b")

# Danh sách các model Groq được hỗ trợ (có thể thay đổi)
SUPPORTED_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.1-405b-reasoning",
    "llama-3-8b-instant",
    "llama-3-70b-8192",
    "mixtral-8x7b-32768",
    "mixtral-8x22b-instruct",
    "gemma-7b-it",
    "gemma2-9b-it",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-guard-4-12b",  # WARNING: Moderation model, không phù hợp cho fact-checking
]

# Models không phù hợp cho fact-checking (moderation/safety models)
MODERATION_MODELS = [
    "meta-llama/llama-guard-4-12b",
    "llama-guard",
]


def set_default_ollama_model(model_name: str):
    global DEFAULT_OLLAMA_MODEL
    if model_name:
        DEFAULT_OLLAMA_MODEL = model_name


def get_default_ollama_model():
    return DEFAULT_OLLAMA_MODEL


def set_default_groq_model(model_name: str):
    global DEFAULT_GROQ_MODEL
    if model_name:
        DEFAULT_GROQ_MODEL = model_name


def get_default_groq_model():
    return DEFAULT_GROQ_MODEL


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

def prompt_groq(prompt, model=None, think=True, track_usage=True):
    """Gọi Groq API để tạo phản hồi từ LLM
    
    Args:
        prompt: Prompt text
        model: Model name (default: DEFAULT_GROQ_MODEL)
        think: Whether to use chain-of-thought
        track_usage: Whether to track token usage (default: True)
    
    Returns:
        str: Response content from LLM
    """
    if not model:
        model = DEFAULT_GROQ_MODEL
    
    # Cảnh báo nếu sử dụng moderation model cho fact-checking
    if any(mod_model in model for mod_model in MODERATION_MODELS):
        print(f"\n⚠️  CẢNH BÁO: Model '{model}' là moderation/safety model, không phù hợp cho fact-checking!")
        print("   Moderation models chỉ trả về 'safe'/'unsafe', không trả về 'Supported'/'Refuted'/'Not Enough Evidence'.")
        print("   Đề xuất sử dụng: llama-3.1-8b-instant, llama-3.1-70b-versatile, hoặc mixtral-8x7b-32768\n")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    client = Groq(api_key=groq_api_key)
    
    messages = []
    if not think:
        # Groq không hỗ trợ system message như Ollama, có thể bỏ qua hoặc thêm vào user message
        messages.append({
            'role': 'user',
            'content': prompt + '\n\nNote: Respond directly without thinking step by step.',
        })
    else:
        messages.append({
            'role': 'user',
            'content': prompt,
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except Exception as e:
        # Check if it's a model not found error
        error_str = str(e).lower()
        if 'not found' in error_str or 'model_not_found' in error_str or '404' in error_str:
            error_msg = f"Model '{model}' không tồn tại hoặc bạn không có quyền truy cập.\n"
            error_msg += f"Lỗi từ Groq API: {str(e)}\n\n"
            
            # Gợi ý thử với prefix nếu model không có prefix
            if '/' not in model and not model.startswith('meta-llama/'):
                suggested_model = f"meta-llama/{model}"
                error_msg += f"Gợi ý: Thử với tên đầy đủ '{suggested_model}'\n\n"
            
            error_msg += "Các model được hỗ trợ phổ biến:\n"
            for m in SUPPORTED_GROQ_MODELS:
                error_msg += f"  - {m}\n"
            error_msg += f"\nBạn có thể xem danh sách đầy đủ tại: https://console.groq.com/docs/models"
            raise ValueError(error_msg) from e
        elif '413' in str(e) or 'rate_limit' in error_str or 'too large' in error_str or 'tpm' in error_str:
            # Rate limit error - pass through so caller can handle (e.g., retry with smaller prompt)
            error_msg = f"Rate limit error với model '{model}': {str(e)}\n"
            error_msg += "Gợi ý: Thử với record ngắn hơn hoặc nâng cấp tier tại https://console.groq.com/settings/billing"
            raise RuntimeError(error_msg) from e
        else:
            error_msg = f"Lỗi khi gọi Groq API với model '{model}': {str(e)}"
            raise RuntimeError(error_msg) from e

    output = response.choices[0].message.content

    if '<think>' in output[:20]:
        output = output.split('</think>')[-1]
    
    # Track token usage
    if track_usage:
        try:
            if hasattr(response, 'usage') and response.usage:
                tracker = get_global_tracker()
                # Groq API returns usage with prompt_tokens, completion_tokens, total_tokens
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                if prompt_tokens > 0 or completion_tokens > 0:
                    tracker.add_usage(prompt_tokens, completion_tokens, model)
        except Exception as e:
            # Silently fail if tracking fails, don't break the main flow
            print(f"Warning: Failed to track token usage: {e}")
        
    return output

def prompt_ollama(prompt, model=None, think=True):
    if not model:
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

