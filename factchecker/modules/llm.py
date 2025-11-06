import openai
import os
import ollama
import base64
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------------------------------------------------------------------
# Định nghĩa các lỗi mạng chung (cho cả OpenAI và Gemini)
LLM_RETRY_EXCEPTIONS = (
    openai.APIError, 
    openai.APITimeoutError, 
    openai.RateLimitError,
    genai.errors.APIError, # Lỗi API của Gemini
)

# Khởi tạo Client Gemini toàn cục
GEMINI_CLIENT = None 
try:
    if os.getenv("GEMINI_API_KEY"):
        GEMINI_CLIENT = genai.Client()
    else:
        print("Warning: GEMINI_API_KEY environment variable not found. Gemini calls will fail.")
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Check GEMINI_API_KEY. Error: {e}")

# -------------------------------------------------------------------

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), 
       retry=retry_if_exception_type(LLM_RETRY_EXCEPTIONS), reraise=True)
def prompt_gemini(prompt, model='gemini-2.5-flash', think=True):
    """Gọi Gemini API, ưu tiên cho các tác vụ Fact-Checking."""
    if GEMINI_CLIENT is None:
        raise ValueError("GEMINI_CLIENT chưa được khởi tạo. Vui lòng thiết lập GEMINI_API_KEY.")
        
    # Đã sửa lỗi: Gửi prompt dưới dạng chuỗi (string)
    response = GEMINI_CLIENT.models.generate_content(
        model=model,
        contents=prompt, 
    )
    return response.text

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), 
       retry=retry_if_exception_type(LLM_RETRY_EXCEPTIONS), reraise=True)
def prompt_openai(prompt, model='o4-mini-2025-04-16'):
    """Gọi OpenAI API (dùng cho mục đích dự phòng hoặc so sánh)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY không được tìm thấy.")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def prompt_ollama(prompt, model='qwen2.5:0.5b', think=True):
    """Gọi mô hình Ollama cục bộ."""
    messages = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({'role': 'user', 'content': prompt})

    response = ollama.chat(model=model, messages=messages)
    output = response['message']['content']

    if '<think>' in output:
        output = output.split('</think>', 1)[-1].strip()
        
    return output

def prompt_llm(prompt, model='gemini-2.5-flash', think=True):
    """
    Hàm gọi LLM mặc định. Ưu tiên sử dụng Gemini.
    """
    if model.startswith('gemini'):
        return prompt_gemini(prompt, model=model, think=think)
    elif model.startswith('o4') or model.startswith('gpt'):
        return prompt_openai(prompt, model=model)
    else:
        return prompt_ollama(prompt, model=model, think=think)