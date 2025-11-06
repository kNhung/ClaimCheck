from .llm import prompt_llm
import re
from typing import Tuple, List, Optional

# -------------------------------------------------------------------
# Định nghĩa các hành động hợp lệ (tương tự planning.py)
ACTION_DEFINITIONS = {
    "geolocate": "Xác định quốc gia/địa điểm của ảnh bằng cách cung cấp ID ảnh.",
    "reverse_search": "Thực hiện tìm kiếm ảnh ngược trên web để tìm ảnh tương tự.",
    "web_search": "Thực hiện tìm kiếm web mở với một truy vấn cụ thể.",
    "image_search": "Lấy các hình ảnh liên quan cho một truy vấn.",
    "NONE": "Không cần hành động tìm kiếm nào nữa."
}

VALID_ACTION_TEXT = "\n".join([f"{k}: {v}" for k, v in ACTION_DEFINITIONS.items()])

DEVELOP_PROMPT = """
Hướng dẫn
Bạn vừa thu thập được Bằng chứng mới (xem Record). Hãy phân tích tính đúng/sai của Phát biểu (Claim) dựa trên bằng chứng đã có.

**QUY TẮC PHÁT TRIỂN LẬP LUẬN (REASONING):**
1. **Phân tích Nguồn:** **BẮT BUỘC** đánh giá sơ bộ độ tin cậy của các nguồn mới thu thập (dựa trên tên báo/cơ quan).
2. **Đối chiếu:** Viết suy luận theo từng bước, đối chiếu bằng chứng mới với Claim.
3. **Mâu thuẫn:** Nếu có nhiều bằng chứng, giải thích tại sao một bằng chứng được chọn thay vì bằng chứng mâu thuẫn khác.
4. **Thiếu sót:** Nếu thông tin chưa đủ để kết luận, nêu rõ **dữ liệu nào còn thiếu**.
5. **Độ dài:** Viết 1–3 đoạn; càng ngắn gọn càng tốt. Dẫn link nguồn bằng Markdown.

**QUY TẮC ĐỀ XUẤT HÀNH ĐỘNG (NẾU CẦN):**
- Nếu cần thu thập thêm bằng chứng, đề xuất **Tối đa 2** hành động mới. Nếu không cần, đề xuất **NONE**.
- **BẮT BUỘC:** Đặt toàn bộ các hành động (hoặc từ NONE) trong một khối mã Markdown duy nhất ở **CUỐI** câu trả lời.

Valid Actions:
{valid_actions}

Ví dụ:
web_search("New Zealand Food Bill 2020")
NONE

Record:
{record}

---
Your Analysis (Suy luận):
"""

def extract_action_from_synthesis(response: str) -> Tuple[str, Optional[List[str]]]:
    """
    Trích xuất phần suy luận (analysis) và các hành động được đề xuất từ phản hồi của LLM.
    """
    code_block_match = re.search(r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
    
    analysis = response
    actions = None
    
    if code_block_match:
        action_content = code_block_match.group(1).strip()
        analysis = response[:code_block_match.start()].strip()
        
        action_lines = [
            line.strip() for line in action_content.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        if action_lines:
            if action_lines[0].upper() == "NONE":
                actions = [] # Không cần hành động
            else:
                actions = action_lines
    
    return analysis, actions


def develop(record: str, think: bool = True) -> Tuple[str, Optional[List[str]]]:
    """
    Phát triển lập luận từ bằng chứng và đề xuất các hành động tiếp theo.
    """
    prompt = DEVELOP_PROMPT.format(valid_actions=VALID_ACTION_TEXT, record=record)
    
    # Dùng prompt_llm (ưu tiên Gemini)
    raw_response = prompt_llm(prompt, model='gemini-2.5-flash', think=think)
    
    analysis, actions = extract_action_from_synthesis(raw_response)

    return analysis, actions