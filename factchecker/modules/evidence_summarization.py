from .llm import prompt_ollama

summarize_prompt = """
Hướng dẫn
Bạn vừa thực hiện tìm kiếm web để tìm bằng chứng. Nhiệm vụ hiện tại là tóm tắt Kết quả Tìm kiếm một cách ngắn gọn, tối đa 5 câu, chỉ bao gồm thông tin liên quan đến Phát biểu đang kiểm chứng.
Bao gồm:
- Thông tin có thể hữu ích cho việc kiểm chứng (xem Record).
- Nếu có: ngày phát hành, tác giả hoặc nhà xuất bản (ví dụ cơ quan báo chí) của kết quả tìm kiếm.
Không bao gồm:
- Quảng cáo, header, footer, sidebar, ... 
- Bất kỳ thông tin không liên quan đến Record hoặc Claim.
Quy tắc bổ sung:
- Không thêm thông tin nào ngoài những gì có trong Kết quả Tìm kiếm. Không thêm thông tin không liên quan đến Claim, dù chúng xuất hiện trong kết quả tìm kiếm.
- Nếu Kết quả Tìm kiếm không chứa thông tin liên quan cho việc kiểm chứng, chỉ in duy nhất một từ viết HOA: NONE. Không in thêm gì khác.
- Giữ phong cách viết nhất quán với các Ví dụ.
- Cố gắng lọc thông tin liên quan ngay cả khi kết quả tìm kiếm ở ngôn ngữ khác.

Lưu ý kỹ thuật: Từ khóa NONE phải giữ nguyên (không dịch), viết hoa toàn bộ.

Claim: {claim}

Evidence:
{url}
{search_result}

BẢN GHI:
{record}

In ra TÓM TẮT ngắn gọn thông tin liên quan YÊU CẦU theo QUY TẮC ở trên:
"""

def summarize(claim, search_result, url, record, think=True):
    prompt = summarize_prompt.format(
        claim=claim,
        search_result=search_result,
        record=record,
        url=url
    )
    return prompt_ollama(prompt, think=think)
    