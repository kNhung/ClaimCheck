from .llm import prompt_llm
import re
from typing import Tuple, List, Optional

# -------------------------------------------------------------------

JUDGE_PROMPT = """
Hướng dẫn
Bạn là Trọng tài Tối cao. Nhiệm vụ là đưa ra Phán quyết (Verdict) cuối cùng về Phát biểu (Claim) dựa trên các bằng chứng đã có (xem Record).

**QUY TẮC BẮT BUỘC:**
1. **Tóm tắt Lập luận:** Viết một đoạn (tối đa 3 câu) **tổng hợp** các ý chính và bằng chứng quan trọng nhất từ quá trình kiểm chứng.
2. **Giải thích Quyết định:** Viết một đoạn **giải thích** tại sao nhãn quyết định được chọn lại là lựa chọn phù hợp nhất (dựa trên các Rules).
3. **Phán quyết cuối cùng:** **BẮT BUỘC** đặt nhãn phán quyết cuối cùng (chỉ một từ/cụm từ, ví dụ: Supported, Refuted, Not Enough Evidence) vào khối mã Markdown **duy nhất** ở **CUỐI** câu trả lời.
4. **Cú pháp:** Nhãn phải là một trong các Decision Options, giữ nguyên tiếng Anh.

Decision Options:
{options}

Rules:
{rules}

Record:
{record}

---
Your Judgement:
"""

VERDICT_EXTRACTION_PROMPT = """
Hướng dẫn
Nhiệm vụ của bạn là trích xuất **nhãn phán quyết duy nhất** từ phần Kết luận đã cho.

**QUY TẮC BẮT BUỘC:**
1. Nhãn phải là một trong các Decision Options.
2. BẮT BUỘC đặt nhãn được trích xuất vào một khối mã Markdown duy nhất ở cuối.

Decision Options:
{options}

Rules:
{rules}

Conclusion:
{conclusion}
Extracted Verdict:
"""

def extract_verdict_from_response(response: str, decision_options: str) -> Optional[str]:
    """
    Trích xuất nhãn phán quyết duy nhất từ phản hồi (tìm kiếm trong khối mã).
    """
    code_block_match = re.search(r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
    
    if code_block_match:
        extracted_text = code_block_match.group(1).strip()
        
        valid_options = [opt.strip().lower() for opt in decision_options.split('\n') if opt.strip()]
        verdict = extracted_text.split('\n')[0].strip()
        
        # Chỉ kiểm tra chữ cái đầu tiên (ví dụ: 'Supported')
        if verdict.lower() in valid_options:
            return verdict
        
        return extracted_text # Fallback: Trả về nội dung đã trích xuất nếu không khớp

    # Fallback cho cú pháp cũ (backticks)
    backtick_match = re.search(r'`([^`]+)`', response)
    if backtick_match:
        return backtick_match.group(1).strip()
        
    return None # Không trích xuất được


def judge(record: str, decision_options: str, rules: str = "", think: bool = True) -> str:
    """
    Xác định tính đúng/sai của tuyên bố và trả về cả lập luận và nhãn phán quyết (raw response).
    Dùng Gemini 2.5 Pro cho tác vụ quyết định cuối cùng.
    """
    prompt = JUDGE_PROMPT.format(record=record, options=decision_options, rules=rules)
    
    # ƯU TIÊN PRO CHO JUDGE
    return prompt_llm(prompt, model='gemini-2.5-pro', think=think)


def extract_verdict(conclusion: str, decision_options: str, rules: str = "") -> Optional[str]:
    """
    Trích xuất phán quyết từ phần kết luận (được sử dụng cho các lần thử lại).
    """
    
    # 1. Thử trích xuất trực tiếp từ khối mã (ổn định nhất)
    verdict = extract_verdict_from_response(conclusion, decision_options)
    if verdict:
        return verdict
        
    # 2. Nếu trích xuất thủ công thất bại, gọi LLM lần 2 để trích xuất lại
    prompt = VERDICT_EXTRACTION_PROMPT.format(conclusion=conclusion, options=decision_options, rules=rules)
    
    # Dùng Gemini 2.5 Pro cho tác vụ trích xuất
    raw_extraction = prompt_llm(prompt, model='gemini-2.5-pro', think=False)
    
    # 3. Thử trích xuất từ phản hồi của mô hình trích xuất
    return extract_verdict_from_response(raw_extraction, decision_options)