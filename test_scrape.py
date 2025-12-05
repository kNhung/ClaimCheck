
from factchecker.tools.web_scraper import scrape_url_content_playwright

url = 'https://laodong.vn/xa-hoi/hai-phong-xem-xet-xu-ly-can-bo-vu-vo-chu-tich-phuong-xay-day-nha-khong-phep-1179905.ldo'
scraped_url_text = scrape_url_content_playwright(url) 
print("======Scrape url text (Playwright):")
print(scraped_url_text)