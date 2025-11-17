import re
from typing import Optional
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from ezmm import MultimodalSequence
from markdownify import MarkdownConverter


def md(soup, **kwargs):
    """Converts a BeautifulSoup object into Markdown."""
    return MarkdownConverter(**kwargs).convert_soup(soup)


def postprocess_scraped(text: str) -> str:
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)

    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)

    return text


def scrape_url_content(url: str) -> Optional[MultimodalSequence]:
    """Fallback scraping script with a 15-second timeout."""
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    def _scrape():
        try:
            page = requests.get(url, headers=headers, timeout=5)
            # Handle any request errors
            if page.status_code == 403:
                return None
            elif page.status_code == 404:
                return None
            page.raise_for_status()
            # soup = BeautifulSoup(page.content, 'html.parser') 
            # if soup.article:
            #     soup = soup.article
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