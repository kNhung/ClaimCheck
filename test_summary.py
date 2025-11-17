from factchecker.modules import evidence_summarization
from factchecker.tools import web_scraper
from time import perf_counter
from datetime import datetime
from factchecker.modules.retriver_rav import get_top_evidence, scrape_text

claim="Ngân 98 bị bắt vì bán hàng giả"
url = "https://tienphong.vn/toan-canh-vu-ngan-98-luong-bang-quang-bi-bat-post1790806.tpo"


# start_ts = datetime.now()
# start = perf_counter()
# print(f"[START] {start_ts.isoformat(timespec='seconds')} | Claim: {claim}")

scraped_content = web_scraper.scrape_url_content(url)
# scraped_content = scrape_text(url)
print("Scraped content:\n", scraped_content)
#summary = evidence_summarization.summarize(claim = claim, model_name="qwen2.5:0.5b", search_result = scraped_content, url = url, record = None)
summary = get_top_evidence(claim, scraped_content)
print("=====Summary:\n", summary)

# end = perf_counter()
# end_ts = datetime.now()
# elapsed_s = end - start
# print(f"[END]   {end_ts.isoformat(timespec='seconds')} | Elapsed: {elapsed_s:.2f}s (~{elapsed_s/60:.2f} min)")

