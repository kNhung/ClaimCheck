# Hướng dẫn sử dụng Redis Cache cho ClaimCheck

## Tổng quan
Hệ thống cache Redis được tích hợp để lưu trữ nội dung các bài báo đã crawl, giúp tránh việc crawl lại cùng một URL nhiều lần và tăng tốc độ xử lý.

## Cài đặt Redis

### 1. Cài đặt Redis Server
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# CentOS/RHEL
sudo yum install redis

# macOS (sử dụng Homebrew)
brew install redis

# Khởi động Redis
sudo systemctl start redis-server  # Linux
brew services start redis         # macOS
```

### 2. Cài đặt thư viện Python
```bash
pip install redis==5.2.0
```

## Cấu hình

### Biến môi trường (Environment Variables)
Thêm vào file `.env` hoặc set trực tiếp:

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Để trống nếu không có password
```

### Cấu hình mặc định
- Host: localhost
- Port: 6379
- Database: 0
- Password: None
- Thời gian expire: 24 giờ (86400 giây)

## Cách hoạt động

### Cache Key
- Format: `scrape:{url}`
- Ví dụ: `scrape:https://example.com/article`

### Quy trình
1. Khi cần crawl URL, hệ thống kiểm tra cache trước
2. Nếu có trong cache, trả về nội dung đã lưu
3. Nếu không có, thực hiện crawl và lưu vào cache
4. Cache tự động expire sau 24 giờ

### Lợi ích
- Giảm thời gian xử lý cho các URL đã crawl
- Giảm tải lên các website nguồn
- Tăng độ ổn định khi website nguồn chậm hoặc lỗi

## Sử dụng trong code

```python
from factchecker.tools.cache import get_cache

# Lấy instance cache
cache = get_cache()

# Lưu dữ liệu
cache.set("key", "value", expire=3600)  # Expire sau 1 giờ

# Lấy dữ liệu
data = cache.get("key")

# Kiểm tra tồn tại
if cache.exists("key"):
    print("Key exists")

# Xóa cache
cache.delete("key")
```

## Quản lý Cache

### Xem cache hiện tại
```bash
# Kết nối Redis CLI
redis-cli

# Xem tất cả keys với prefix scrape:
KEYS scrape:*

# Xem nội dung một key
GET "scrape:https://example.com"

# Xóa một key
DEL "scrape:https://example.com"

# Xóa tất cả keys với pattern
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 scrape:*
```

### Thống kê
```bash
# Số lượng keys
DBSIZE

# Thông tin memory
INFO memory
```

## Troubleshooting

### Lỗi kết nối Redis
- Kiểm tra Redis server có chạy: `redis-cli ping`
- Kiểm tra port: `netstat -tlnp | grep 6379`
- Kiểm tra firewall

### Cache không hoạt động
- Nếu Redis không khả dụng, hệ thống sẽ tự động bỏ qua cache
- Kiểm tra logs để xem warning về kết nối Redis

### Cache quá lớn
- Điều chỉnh thời gian expire trong code
- Dọn dẹp cache định kỳ
- Sử dụng Redis persistence nếu cần lưu lâu dài

## Mở rộng

### Multiple Databases
Có thể sử dụng các database khác nhau cho các mục đích khác nhau:
- DB 0: Cache scraping
- DB 1: Cache search results
- DB 2: Cache processed data

### Cluster Redis
Cho môi trường production với nhiều instances.

### Redis với Docker
```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```