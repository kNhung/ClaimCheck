from .llm import prompt_ollama


plan_prompt = """
Nhiệm vụ: Lên kế hoạch tìm thêm bằng chứng để kiểm chứng phát biểu (Claim).
Claim: {claim}
Bạn CHỈ ĐƯỢC phép in ra danh sách các hành động theo đúng cú pháp bên dưới.
Mỗi hành động phải được in trong cùng một khối mã (```) duy nhất.
Không được giải thích, không được in văn bản tự nhiên, không thêm mô tả.

Quy tắc:
- Hành động đầu tiên *bắt buộc* phải là:
  web_search("{claim}")
- Không thêm văn bản khác ngoài khối mã hành động.
- Không được thêm link thủ công hoặc hướng dẫn.

Hành động hợp lệ:
{valid_actions}

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (thay ... bằng từ, cụm từ, hoặc thực thể từ Yêu cầu):
{examples}

BẢN GHI:
{record}

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
    
    if not actions:
        actions = ["web_search"]
        
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    
    prompt = plan_prompt.format( 
        valid_actions=valid_actions, 
        examples=examples, 
        record=record, 
        claim=claim,
    ) 
    response = prompt_ollama(prompt, think=think) 
    return response
