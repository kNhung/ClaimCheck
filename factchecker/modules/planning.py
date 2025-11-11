from .llm import prompt_model

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

def plan(claim, model_name, record="", examples="", actions=None, think=True, key_number=1):
    action_definitions = {
        "geolocate": {"desc": "Xác định quốc gia hoặc địa điểm chụp của hình ảnh, nếu hình ảnh có cảnh vật hoặc địa danh.", "example": "geolocate(<image:k>)"},
        "reverse_search": {"desc": "Tìm ngược hình ảnh trên web để kiểm chứng nguồn gốc hoặc phát hiện hình ảnh tương tự.", "example": "reverse_search(<image:k>)"},
        "web_search": {"desc": "Tìm kiếm trên web thông tin hoặc bài báo chính thống giúp xác minh hoặc bác bỏ yêu cầu.", "example": 'web_search("...")'},
        "image_search": {"desc": "Tìm hình ảnh liên quan để đối chiếu hoặc xác minh nội dung hình ảnh trong yêu cầu.", "example": 'image_search("...")'},
    }
    if not actions:
        actions = ["web_search", "image_search"]
    elif actions == "All":
        actions = list(action_definitions.keys())
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    return prompt_model(prompt, model_name=model_name, think=think, key_number=key_number)