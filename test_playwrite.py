from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        # headless=True là mặc định, nhưng viết rõ để chắc chắn
        browser = p.chromium.launch(headless=True) 
        page = browser.new_page()
        
        print("Đang truy cập...")
        page.goto("https://laodong.vn/xa-hoi/hai-phong-xem-xet-xu-ly-can-bo-vu-vo-chu-tich-phuong-xay-day-nha-khong-phep-1179905.ldo")
        
        # Chờ nội dung load (thay thế time.sleep tốt hơn)
        page.wait_for_selector("h1")
        
        title = page.title()
        print(f"Tiêu đề trang web: {title}")
        
        # Chụp ảnh bằng chứng (debug trên server rất tiện)
        page.screenshot(path="screenshot.png")
        
        browser.close()

if __name__ == "__main__":
    run()