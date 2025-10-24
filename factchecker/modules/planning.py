from .llm import prompt_ollama

plan_prompt = """Hướng dẫn
Kiến thức hiện có chưa đủ để đánh giá tính đúng sai của Phát biểu (Claim).
Vì vậy, hãy đề xuất một tập hành động để thu thập thêm bằng chứng hữu ích. Tuân thủ các quy tắc sau:
- Các hành động hợp lệ được liệt kê trong phần Valid Actions (kèm mô tả ngắn). Hiện tại không có hành động nào khác ngoài danh sách này.
- Với mỗi hành động, phải dùng đúng định dạng như trong Valid Actions.
- Đặt toàn bộ các hành động trong một khối mã (Markdown code block) duy nhất ở cuối câu trả lời.
- Đề xuất càng ít hành động càng tốt nhưng đủ cần thiết. Không đề xuất các hành động trùng lặp hoặc đã dùng rồi.
- Cân bằng giữa các phương thức (nếu có); tuy nhiên vẫn cần kiểm chứng đúng/sai của phát biểu văn bản.
- So sánh hình ảnh và chú thích (nếu liên quan) để xác thực ngữ cảnh.

Lưu ý kỹ thuật: Giữ nguyên chính xác cú pháp hành động khi xuất ra, ví dụ web_search("...") và từ khóa NONE khi không có hành động. Không dịch hoặc thay đổi "web_search" hay "NONE".

Valid Actions:
{valid_actions}

Ví dụ:
{examples}

Record:
{record}

Claim: {claim}
Your Actions:
"""

decompose_prompt = """Hướng dẫn
Phân rã phát biểu thành các tiểu phát biểu/câu hỏi nhỏ, cụ thể và có thể xử lý độc lập. Số lượng không quá 5.

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
