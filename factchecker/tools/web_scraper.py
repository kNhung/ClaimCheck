import os
import re
from typing import Optional
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from ezmm import MultimodalSequence
from markdownify import MarkdownConverter

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Optional


def md(soup, **kwargs):
    """Converts a BeautifulSoup object into Markdown."""
    return MarkdownConverter(**kwargs).convert_soup(soup)


def postprocess_scraped(text: str) -> str:
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)

    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)

    return text


SCRAPE_TIMEOUT = float(os.getenv("FACTCHECKER_SCRAPE_TIMEOUT", "4"))


def scrape_url_content(url: str) -> Optional[MultimodalSequence]:
    """Fallback scraping script with a 15-second timeout."""
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    def _scrape():
        try:
            page = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
            # Handle any request errors
            if page.status_code == 403:
                return None
            elif page.status_code == 404:
                return None
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs])
            #text = md(text)
            text = postprocess_scraped(text)
            return text
        except requests.exceptions.RequestException:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_scrape)
        try:
            result = future.result(timeout=15)
            if result is None:
                return None
            return result
        except concurrent.futures.TimeoutError:
            return "Unable to Scrape"


def scrape_url_content_playwright(url: str) -> Optional[str]:
    """Scrape using Playwright for JS-rendered content."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait a bit more for JS
            page.wait_for_timeout(3000)
            
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            browser.close()
            
            paragraphs = soup.find_all('p')
            if paragraphs:
                text = " ".join([p.get_text() for p in paragraphs])
            else:
                content_divs = soup.find_all('div', class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower()))
                if content_divs:
                    text = " ".join([div.get_text() for div in content_divs])
                else:
                    text = soup.body.get_text() if soup.body else ""
            
            text = postprocess_scraped(text)
            return text if text.strip() else None
    except Exception as e:
        print(f"Playwright scrape error: {e}")
        return None
    

def scrape_url_content_playwright(url: str) -> Optional[str]: # Đã đổi return type hint cho rõ nghĩa
    """
    Playwright scraping script with a 15-second timeout.
    Optimized for Ubuntu/Linux environment.
    """
    SCRAPE_TIMEOUT = 15000  # Playwright tính bằng milliseconds (15s = 15000ms)
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    with sync_playwright() as p:
        # Launch browser
        # args=['--no-sandbox'] là BẮT BUỘC nếu chạy dưới quyền root hoặc trong Docker trên Ubuntu
        browser = p.chromium.launch(
            headless=True, 
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        try:
            # Tạo context để gán User-Agent
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            
            # Set timeout chung cho các thao tác
            page.set_default_timeout(SCRAPE_TIMEOUT)

            # Truy cập trang web
            # wait_until='domcontentloaded': Đợi HTML tải xong (nhanh hơn networkidle)
            # Nếu trang quá nặng JS, hãy đổi thành 'networkidle'
            response = page.goto(url, wait_until="domcontentloaded")

            # Xử lý các lỗi HTTP (tương đương requests)
            if not response:
                return None
            
            status_code = response.status
            if status_code == 403 or status_code == 404:
                return None
            if not response.ok: # Check các lỗi 4xx 5xx khác
                return None

            # Logic lấy thẻ <p> tương đương BeautifulSoup
            # locator('p').all_inner_texts() sẽ lấy text của tất cả thẻ p và trả về list
            paragraphs = page.locator('p').all_inner_texts()
            text = " ".join(paragraphs)

            # Gọi hàm xử lý hậu kỳ của bạn
            # Giả định hàm postprocess_scraped đã được định nghĩa ở ngoài
            text = postprocess_scraped(text)
            
            return text

        except PlaywrightTimeoutError:
            return "Unable to Scrape"
        except Exception as e:
            # print(f"General Error: {e}") # Uncomment để debug nếu cần
            return None
        finally:
            # Luôn đóng browser để giải phóng RAM trên Ubuntu
            browser.close()