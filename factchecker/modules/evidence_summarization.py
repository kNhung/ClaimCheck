from .llm import prompt_groq

summarize_prompt = """HƯỚNG DẪN
Bạn là trợ lý kiểm chứng thông tin. Nhiệm vụ của bạn là đọc KẾT QUẢ TÌM KIẾM và TÓM TẮT ngắn gọn những thông tin LIÊN QUAN đến YÊU CẦU dưới đây.

ĐỊNH NGHĨA:
"Tóm tắt" có nghĩa là trích ra và diễn đạt lại NGẮN GỌN các thông tin CHÍNH có LIÊN QUAN đến YÊU CẦU, 
mà không thêm bất kỳ nhận xét, phán đoán, hay thông tin ngoài nội dung gốc.

MỤC TIÊU:
- Chỉ chọn các thông tin giúp xác định xem YÊU CẦU là đúng hay sai.

QUY TẮC:
- Tóm tắt trong tối đa 5 câu ngắn gọn, tập trung vào phần LIÊN QUAN TRỰC TIẾP đến YÊU CẦU.
- Nếu phát hiện bằng chứng xác nhận hoặc phủ nhận, phải nêu rõ điều đó trong tóm tắt.
- Nếu không có thông tin nào liên quan, chỉ in đúng một từ: NONE.
- KHÔNG thêm nhận xét, phân tích, quảng cáo hoặc nội dung không liên quan đến YÊU CẦU.

YÊU CẦU: {claim}

KẾT QUẢ TÌM KIẾM:
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
    return prompt_groq(prompt, think=think)
    