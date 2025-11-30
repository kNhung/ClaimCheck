from .llm import prompt_ollama

plan_prompt = """HƯỚNG DẪN
Mọi phát ngôn đều phải có thể kiểm chứng và dựa trên dữ liệu đã được khai thác. Nếu không đủ thông tin, phải kết luận `Not Enough Information`. Tuyệt đối không được giả định.
GIAO THỨC KHÔNG ẢO GIÁC: trước khi trả lời, cần kiểm tra lại tính xác thực của thông tin.

Nhiệm vụ: Tìm bằng chứng để kiểm chứng phát biểu (Claim).
Claim: {claim}

Quy tắc:
- THỰC HIỆN TỐI THIỂU 2 HÀNH ĐỘNG.
- Bạn CHỈ ĐƯỢC phép in ra danh sách các hành động theo đúng cú pháp bên dưới.
- Mỗi hành động phải được in trong cùng một khối mã (```) duy nhất.
- Không in ra danh sách các bước.
- Không được giải thích, không được in văn bản tự nhiên, không thêm mô tả.


Hành động hợp lệ:
web_search("{claim}")
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
            #"example": 'web_search("{claim}")',
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
    #examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    response = prompt_ollama(prompt, think=think)
    # return response

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