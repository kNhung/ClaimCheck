from .llm import prompt_ollama

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

YÊU CẦU: {claim}

In ngắn gọn đề xuất của bạn, chỉ có hành động theo ĐỊNH DẠNG ĐẦU RA BẮT BUỘC, mỗi hành động nằm trên 1 dòng:
"""

decompose_prompt = """Instructions
Decompose the claim into smaller, manageable sub-claims or questions that can be addressed individually. Each sub-claim should be specific and focused.
There should be no more than 5 sub-claims.

Claim: {claim}
Your Sub-Claims:

"""

def plan(claim, record="", examples="", actions=None, think=True):
    action_definitions = {
        "geolocate": {"desc": "Xác định quốc gia nơi ảnh được chụp bằng cách cung cấp ID ảnh.", "example": "geolocate(<image:k>)"},
        "reverse_search": {"desc": "Thực hiện tìm kiếm ảnh ngược trên web để tìm ảnh tương tự.", "example": "reverse_search(<image:k>)"},
        "web_search": {"desc": "Thực hiện tìm kiếm web mở cho các trang liên quan.", "example": 'web_search("New Zealand Food Bill 2020")'},
        "image_search": {"desc": "Lấy các hình ảnh liên quan cho một truy vấn.", "example": 'image_search("China officials white suits carry people")'}
    }
    if not actions:
        actions = ["web_search", "image_search"]
    elif actions == "All":
        actions = list(action_definitions.keys())
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    response = prompt_ollama(prompt, think=think)
    return response
