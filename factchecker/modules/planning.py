from .llm import prompt_ollama

plan_prompt = """
Nhiệm vụ: Lên kế hoạch tìm thêm bằng chứng để kiểm chứng phát biểu (Claim).

Chỉ được dùng các hành động sau (đúng cú pháp):
- web_search("..."): tìm trang liên quan.
- NONE: không cần hành động.

Quy tắc:
- Đề xuất đủ hành động, không trùng lặp, không ngoài danh sách.
- In các hành động trong một khối mã (```) duy nhất ở cuối.

Record:
{record}
Claim: {claim}

Your Actions:
"""
decompose_prompt = """Hướng dẫn
Phân tách phát biểu dưới đây thành 1-3 **tiểu phát biểu** có thể được kiểm chứng độc lập, giữ ngữ nghĩa gốc, tránh suy diễn hoặc mở rộng nội dung.


Claim: {claim}
Your Sub-Claims:

"""

def plan(claim, record="", examples="", actions=None, think=True):
    action_definitions = {
        "web_search": {
            "desc": "Thực hiện tìm kiếm web mở cho các trang liên quan.", 
            "example": 'web_search("New Zealand Food Bill 2020")',
            "params": {
                "query": "Từ khóa tìm kiếm"
            }
        }
    }
    
    decompose_text = decompose_prompt.format(claim=claim)
    subclaims_response = prompt_ollama(decompose_text, think=think)
    subclaims = [] 
    
    for line in subclaims_response.splitlines():
        line = line.strip()
        if line:
            subclaims.append(line)
    
    if not subclaims:
        subclaims = [claim]
        
    if not actions:
        actions = ["web_search"]
        
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    
    all_actions = []

    for sub in subclaims:
        prompt = plan_prompt.format(
            valid_actions=valid_actions, 
            examples=examples,
            record=record,
            claim=sub
        )
        response = prompt_ollama(prompt, think=think)
        all_actions.extend(response.splitlines())
    return "\n".join(all_actions)

