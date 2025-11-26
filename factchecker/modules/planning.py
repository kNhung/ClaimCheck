from .llm import prompt_ollama

plan_prompt = """HƯỚNG DẪN
Kiến thức hiện tại vẫn chưa đủ để đánh giá tính xác thực của YÊU CẦU.
Hãy in ra CÂU TÌM KIẾM để thu thập bằng chứng mới, theo đúng quy tắc sau:

### QUY TẮC:
- CÂU TÌM KIẾM là một cụm từ hoặc câu ngắn gọn dùng để tìm kiếm thông tin trên web.
- CÂU TÌM KIẾM liên quan trực tiếp đến YÊU CẦU và chứa ít nhất một từ khóa hoặc thực thể trong YÊU CẦU.

### YÊU CẦU: 
{claim}

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ in CÂU TÌM KIẾM, KHÔNG in thêm mô tả hay giải thích.

In ra câu tìm kiếm của bạn:
"""

def plan(claim, think=True):
    prompt = plan_prompt.format(claim=claim)
    response = prompt_ollama(prompt, think=think)
    return response
