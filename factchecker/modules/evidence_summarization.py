from .llm import prompt_llm 
from typing import Dict, Any, Optional
import re

# 💡 Cải tiến: Sử dụng block Markdown để định dạng đầu ra Summary
SUMMARIZE_PROMPT = """
Hướng dẫn
Bạn vừa thực hiện tìm kiếm web để tìm bằng chứng cho Phát biểu (Claim). Nhiệm vụ là **tóm tắt** Kết quả Tìm kiếm này.

**QUY TẮC BẮT BUỘC:**
1. **Ngắn gọn:** Tóm tắt tối đa **4 câu**.
2. **Liên quan:** CHỈ bao gồm sự kiện **liên quan trực tiếp** đến Phát biểu đang kiểm chứng.
3. **Đánh giá Nguồn:** **BẮT BUỘC** trích dẫn: **Ngày phát hành** và **Tên cơ quan/báo chí** đã đăng tải thông tin (nếu có trong Content), để đánh giá độ tin cậy.
4. **Không thêm:** KHÔNG thêm thông tin nào ngoài những gì có trong Content, KHÔNG thêm ý kiến cá nhân.
5. **Đầu ra Dạng Khối:** BẮT BUỘC đặt toàn bộ bản tóm tắt (hoặc từ NONE) trong một khối mã Markdown duy nhất ở cuối câu trả lời.
6. **Không liên quan (Fallthrough):** Nếu Content không chứa thông tin liên quan, chỉ in **duy nhất một từ viết HOA** trong khối mã: **NONE**.

Lưu ý kỹ thuật: Từ khóa NONE phải giữ nguyên (không dịch), viết hoa toàn bộ.

Claim: {claim}

---
Evidence Source:
URL: {url}
Content:
{search_result}

Record:
{record}

Your Summary:
"""

def summarize(claim: str, search_result: str, url: str, record: str, think: bool = True) -> str:
    """
    Tạo bản tóm tắt bằng chứng từ kết quả tìm kiếm web.
    """
    
    MAX_SEARCH_CONTENT = 6000 # Giới hạn 6000 ký tự cho nội dung tìm kiếm
    
    prompt = SUMMARIZE_PROMPT.format(
        claim=claim,
        search_result=search_result[:MAX_SEARCH_CONTENT],
        record=record,
        url=url
    )
    
    # Dùng prompt_llm (ưu tiên Gemini)
    raw_response = prompt_llm(prompt, model='gemini-2.5-flash', think=think)
    
    return raw_response