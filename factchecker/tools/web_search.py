import json
import os
from datetime import datetime
import requests

# Load environment variables from a .env file if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # It's okay if python-dotenv isn't installed; we'll rely on real env vars.
    pass

def web_search(query, date, top_k=3, **kwargs):
    """
    Fetches search results using the Serper API and extracts URLs and snippets.

    Parameters:
    query (str): The search query.
    date (datetime.date): The date to filter results up to.
    top_k (int): The number of search results to fetch.

    Returns:
    tuple: A tuple containing two lists:
           - List of URLs from the organic results.
           - List of snippets from the organic results (empty string if snippet is missing).
    """
    # Format the date to the required format
    end_date = datetime.strptime(date, "%d-%m-%Y").strftime('%d/%m/%Y')

    # Đọc API key và CX từ biến môi trường
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")

    if not google_api_key or not google_cx:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY or GOOGLE_CX. "
            "Set them as environment variables or in a .env file."
        )

    # Google Custom Search endpoint
    url = "https://www.googleapis.com/customsearch/v1"

    # Tạo payload query
    params = {
        "key": google_api_key,
        "cx": google_cx,
        "q": query,
        "num": min(top_k, 10),  # Google cho phép tối đa 10 mỗi lần
        "lr": "lang_vi",        # Ưu tiên tiếng Việt
        "safe": "active"        # Ẩn nội dung không an toàn
    }

    # Gửi request
    response = requests.get(url, params=params)

    # Kiểm tra lỗi
    if response.status_code != 200:
        raise RuntimeError(f"Google API error ({response.status_code}): {response.text}")

    # Parse kết quả JSON
    data = response.json()

    # Lấy danh sách kết quả gọn gàng
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })
    return results