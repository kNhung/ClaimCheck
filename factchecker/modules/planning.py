from .llm import prompt_groq

plan_prompt = """HƯỚNG DẪN
Kiến thức hiện tại vẫn chưa đủ để đánh giá tính xác thực của YÊU CẦU.
Hãy đề xuất ngắn gọn các hành động để thu thập bằng chứng mới, theo đúng quy tắc sau:

QUY TẮC:
- Mỗi hành động là 1 cụm từ được liệt kê trong mục HÀNH ĐỘNG HỢP LỆ, đi cùng là mô tả sau dấu ":". Hành động được dùng theo định dạng trong mục ĐỊNH DẠNG ĐẦU RA BẮT BUỘC.
- Đề xuất hành động phải liên quan trực tiếp đến Yêu cầu và chứa ít nhất một từ khóa hoặc thực thể trong Yêu cầu.
- Không đề xuất các hành động tương tự hoặc đã được sử dụng trước đó trong BẢN GHI.
- KHÔNG in bất cứ thứ gì ngoài các Hành động đề xuất.

HÀNH ĐỘNG HỢP LỆ (Hành động: mô tả):
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
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    response = prompt_groq(prompt, think=think)
    return response
