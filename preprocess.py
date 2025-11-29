import re

def preprocess(text):
    # 1. Xóa emoji và ký tự không phải chữ/số/dấu câu cơ bản
    text = re.sub(r'[^\w\s\.,\-:/]', '', text)
    
    # 2. Thêm khoảng trắng giữa chữ và số nếu dính nhau
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # 3. Xóa khoảng trắng dư
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Xóa khoảng trắng đầu/cuối
    return text.strip()
