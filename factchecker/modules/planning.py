# SỬA LỖI: Thay thế import prompt_ollama bằng prompt_llm
from .llm import prompt_llm 
import re
from typing import List

# -------------------------------------------------------------------
# Định nghĩa các hành động hợp lệ (có thêm NONE)
ACTION_DEFINITIONS = {
    "geolocate": {"desc": "Xác định quốc gia/địa điểm của ảnh bằng cách cung cấp ID ảnh.", "example": "geolocate(<image:k>)"},
    "reverse_search": {"desc": "Thực hiện tìm kiếm ảnh ngược trên web để tìm ảnh tương tự.", "example": "reverse_search(<image:k>)"},
    "web_search": {"desc": "Thực hiện tìm kiếm web mở với một truy vấn cụ thể.", "example": 'web_search("New Zealand Food Bill 2020")'},
    "image_search": {"desc": "Lấy các hình ảnh liên quan cho một truy vấn.", "example": 'image_search("China officials white suits carry people")'},
    "NONE": {"desc": "Không cần hành động tìm kiếm nào nữa.", "example": "NONE"}
}
# -------------------------------------------------------------------

PLAN_PROMPT = """Hướng dẫn
Kiến thức hiện có chưa đủ để đánh giá tính đúng sai của Phát biểu (Claim).
Vì vậy, hãy đề xuất một tập hành động để thu thập thêm bằng chứng hữu ích.

**QUY TẮC BẮT BUỘC:**
1. **Số lượng:** Đề xuất tối đa **2 hành động** cần thiết nhất. Không đề xuất các hành động trùng lặp hoặc đã dùng rồi.
2. **Chi tiết Truy vấn:** Nếu là web_search hoặc image_search, truy vấn phải **cụ thể**, bao gồm **TÊN RIÊNG** (tổ chức, người, địa điểm) và **NGÀY THÁNG/THỜI GIAN** (nếu có trong Claim).
3. **Cú pháp:** Với mỗi hành động, phải dùng đúng định dạng và cú pháp như trong Valid Actions.
4. **Khối mã:** **Bắt buộc** đặt toàn bộ các hành động **HỢP LỆ** (hoặc NONE) trong một khối mã Markdown duy nhất, nằm ở **CUỐI** câu trả lời.
5. **Không dịch/Thay đổi:** Giữ nguyên chính xác "web_search", "geolocate", "NONE", v.v.

Valid Actions:
{valid_actions}

Ví dụ:
{examples}

---
Record:
{record}

Claim: {claim}

Your Actions (BẮT BUỘC SỬ DỤNG CÚ PHÁP KHỐI MÃ MARKDOWN):
"""

DECOMPOSE_PROMPT = """Hướng dẫn
Phân rã phát biểu thành các tiểu phát biểu/câu hỏi nhỏ, cụ thể và có thể xử lý độc lập. Số lượng không quá 5.

Claim: {claim}
Your Sub-Claims:

"""

def extract_actions_from_response(response: str) -> List[str]:
    """
    Trích xuất các hành động hợp lệ từ khối mã Markdown trong phản hồi của LLM.
    """
    code_block_match = re.search(r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
    
    if code_block_match:
        code_block = code_block_match.group(1).strip()
        
        action_lines = [
            line.strip() for line in code_block.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ]
        return action_lines
    return []


def plan(claim, record="", think=True, actions: List[str] = None):
    """
    Tạo kế hoạch hành động để tìm bằng chứng cho một tuyên bố.
    """
    if actions is None:
        actions_to_use = ["web_search", "NONE"] 
    elif actions == "All":
        actions_to_use = list(ACTION_DEFINITIONS.keys())
    else:
        actions_to_use = actions
        
    valid_actions = "\n".join([f"{a}: {ACTION_DEFINITIONS[a]['desc']}" for a in actions_to_use])
    examples = "\n".join([f"{ACTION_DEFINITIONS[a]['example']}" for a in actions_to_use if a != "NONE"])

    prompt = PLAN_PROMPT.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    
    # Dùng prompt_llm (ưu tiên Gemini)
    raw_response = prompt_llm(prompt, model='gemini-2.5-flash', think=think)
    
    return extract_actions_from_response(raw_response)

def decompose(claim):
    """
    Phân rã tuyên bố thành các tiểu tuyên bố.
    """
    prompt = DECOMPOSE_PROMPT.format(claim=claim)
    # Dùng prompt_llm (ưu tiên Gemini)
    response = prompt_llm(prompt, model='gemini-2.5-flash', think=False)
    return response