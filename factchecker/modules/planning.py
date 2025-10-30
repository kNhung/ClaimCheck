from .llm import prompt_ollama

plan_prompt = """
Nhiệm vụ: Lên kế hoạch tìm thêm bằng chứng để kiểm chứng phát biểu (Claim).

Chỉ được dùng các hành động sau (đúng cú pháp):
- geolocate(<image:k>): xác định vị trí ảnh.
- reverse_search(<image:k>): tìm ảnh tương tự.
- web_search("..."): tìm trang liên quan.
- image_search("..."): tìm hình ảnh liên quan.
- NONE: không cần hành động.

Quy tắc:
- Đề xuất ít, đủ, không trùng lặp, không ngoài danh sách.
- In các hành động trong một khối mã (```) duy nhất ở cuối.

Record:
{record}
Claim: {claim}

Your Actions:
"""
decompose_prompt = """Hướng dẫn
Phân tách phát biểu dưới đây thành 1-3 **tiểu phát biểu** có thể được kiểm chứng độc lập, giữ ngữ nghĩa gốc, tránh suy diễn hoặc mở rộng nội dụng


Claim: {claim}
Your Sub-Claims:

"""

def plan(claim, record="", examples="", actions=None, think=True):
    action_definitions = {
        "geolocate": {
            "desc": "Xác định địa danh, quốc gia nơi ảnh được chụp bằng cách cung cấp ID ảnh.", 
            "example": "geolocate(<image:k>)",
            "params": {
                "image": "ID ảnh"
            }
        },
        "reverse_search": {
            "desc": "Thực hiện tìm kiếm ảnh ngược trên web để tìm ảnh tương tự.", 
            "example": "reverse_search(<image:k>)",
            "params": {
                "image": "ID ảnh"
            }
        },
        "web_search": {
            "desc": "Thực hiện tìm kiếm web mở cho các trang liên quan.", 
            "example": 'web_search("New Zealand Food Bill 2020")',
            "params": {
                "query": "Từ khóa tìm kiếm"
            }
        },
        "image_search": {
            "desc": "Lấy các hình ảnh liên quan cho một truy vấn.", 
            "example": 'image_search("China officials white suits carry people")',
            "params": {
                "query": "Từ khóa tìm kiếm"
            }
        }
    }
    if not actions:
        actions = ["web_search", "image_search"]
    elif actions == "All":
        actions = list(action_definitions.keys())
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(
        valid_actions=valid_actions, 
        examples=examples,
        record=record,
        claim=claim
    )
    response = prompt_ollama(prompt, think=think)
    return response