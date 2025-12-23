# Hướng Dẫn Triển Khai (Deployment Guide)

## Tổng Quan Kiến Trúc

Hệ thống ClaimCheck được triển khai 2 thành phần chính:

### Backend Service (FastAPI)
- **Container**: `claimcheck-backend`
- **Port**: `8000`
- **Framework**: FastAPI với Uvicorn
- **Chức năng**: 
  - API xử lý fact-checking (`/factcheck/verify`)
  - Quản lý reports (`/reports`)
  - Health checks (`/health`)
  - API Documentation (`/docs`, `/redoc`)

### Frontend Service (React + Nginx)
- **Container**: `claimcheck-frontend`
- **Port**: `80`
- **Framework**: React (Vite) + Nginx
- **Chức năng**: 
  - Giao diện web cho người dùng
  - Proxy API requests đến backend qua `/api`
  - Serve static files

### Lưu Trữ Dữ Liệu
- **Reports**: Được lưu trong thư mục `./reports` trên host, được mount vào container
- **Format**: Mỗi report bao gồm `report.json`, `report.md`, `evidence.md`

---

### Pipeline (Luồng xử lý nhanh)

Tổng quan ngắn về luồng xử lý claim trong hệ thống và các file chính tương ứng:

- Bước 0. Tiền xử lý & phân đoạn: chuẩn hoá văn bản, phục hồi dấu tiếng Việt, loại bỏ ký tự rác.
  - File chính: `factchecker/preprocessing/preprocessing.py`

- Bước 1. Phát hiện & lọc claim: tách câu, xác định câu nào đủ điều kiện để fact-check.
  - File chính: `factchecker/modules/claim_detection.py`

- Bước 2. Lập kế hoạch truy vấn (Claim → Query): sinh các truy vấn/search queries để thu thập bằng chứng.
  - File chính: `factchecker/modules/planning.py`

- Bước 3. Web Search: tìm các URL/snippet liên quan với query.
  - File chính: `factchecker/tools/web_search.py`

- Bước 4. Web Scraping: lấy nội dung trang (HTML → plain text) để làm nguồn evidence.
  - File chính: `factchecker/tools/web_scraper.py`
  - File phụ (cache bài báo đã scrape để dùng lại): `factchecker/tools/cache.py`

- Bước 5. Retriever + Ranker (RAV): chunk hoá văn bản, embedding (bi-encoder) để lấy top-p, sau đó re-rank bằng cross-encoder lấy top-q evidence.
  - File chính: `factchecker/modules/retriver_rav.py`

- Bước 6. Kiểm tra tính đầy đủ của bằng chứng (nếu enable): kiểm tra đã đủ bằng chứng chưa, nếu chưa thì đề xuất hành động để tìm thêm bằng chứng.
  - File chính: `factchecker/modules/evidence_synthesis.py`

- Bước 7. Judge (LLM): dùng model (Ollama / Gemini) để đưa ra verdict (Supported / Refuted / Not Enough Evidence) và giải thích.
  - File chính:  `factchecker/modules/evaluation.py` (nếu cần chuẩn hoá input)

- Orchestrator & Report: logic điều phối toàn bộ pipeline, logging và ghi report vào thư mục `reports/`.
  - File chính: `factchecker/factchecker.py` (điều phối) và `fact-check.py` (script runner để chạy batch)

Ghi chú: các bước trên được ghi log chi tiết vào `factchecker/report/report_writer.py` và có thể tuỳ biến bằng biến môi trường (ví dụ `FACTCHECKER_MAX_ACTIONS`, `FACTCHECKER_MODEL_NAME`, `FACTCHECKER_EMBED_DEVICE`).

### Cài Đặt Docker (nếu chưa có)
```bash
# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Thêm user vào group docker để chạy không cần sudo
sudo usermod -aG docker $USER
newgrp docker

# Cài đặt Docker Compose plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

### Cài Đặt Redis 

- Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
sudo systemctl status redis-server
```

- macOS (Homebrew):
```bash
brew install redis
brew services start redis
```

- Chạy nhanh bằng Docker (mapping port 6379):
```bash
docker run -p 6379:6379 -d --name claimcheck-redis redis:7
```

- Lưu ý:
  - Thư viện Python `redis` đã được liệt kê trong `requirements.txt`; đảm bảo service Redis đang chạy trước khi sử dụng cache.
  - Khởi động Redis cục bộ:
```bash
redis-server
# Hoặc chạy nền:
redis-server --daemonize yes
```

- Tham số kết nối (env): `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`.

- Xem chi tiết cấu hình và hướng dẫn sử dụng cache trong `REDIS_CACHE_GUIDE.md`.

## Chuẩn Bị Biến Môi Trường

### 1. Tạo File `.env`

Copy .env.example sang file `.env` trong thư mục gốc của project (`ClaimCheck/`):

```bash
cd /path/to/ClaimCheck
cp .env.example .env
```

### 2. Cấu Hình Các Biến Môi Trường

Sau khi copy file `.env.example` sang `.env`, mở file `.env` và điền các giá trị thực tế.

**Các biến bắt buộc:**
- `SERPER_API_KEY`: API key từ Serper (bắt buộc)

**Các biến chính trong `.env.example` (mặc định có trong file):**
- `FACTCHECKER_BI_ENCODER`: Tên bi-encoder cho embedding (mặc định: `paraphrase-multilingual-MiniLM-L12-v2`)
- `FACTCHECKER_CROSS_ENCODER`: Tên cross-encoder cho reranking (mặc định: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `FACTCHECKER_JUDGE_PROVIDER`: Provider cho bước judge (`gemini` hoặc `ollama`, mặc định `gemini`).

**Khi dùng Gemini làm judge (cần thiết nếu `FACTCHECKER_JUDGE_PROVIDER=gemini`)**
- `GEMINI_API_KEY`: API key từ Google (bắt buộc nếu dùng Gemini)
- `GEMINI_MODEL`: Tên model Gemini (mặc định: `gemini-2.5-flash`)

**Khi dùng Ollama / Qwen (cấu hình local):**
- `OLLAMA_JUDGE_MODEL`: (tùy chọn) model Ollama dành cho judging (ví dụ: `qwen2.5:3b`) — thường được ghi comment trong `.env.example`.
- `OLLAMA_MODEL_NAME`: model Ollama mặc định cho các bước LLM (mặc định: `qwen2.5:1.5b`).

**Các biến cấu hình khác:**
- `FACTCHECKER_MAX_ACTIONS`: Số lượng actions tối đa khi lập kế hoạch (mặc định: `2`).
- `FACTCHECKER_EMBED_DEVICE`: Thiết bị cho embedding (`cuda` hoặc `cpu`). `.env.example` mặc định là `cuda` (hệ thống sẽ fallback về `cpu` nếu GPU không khả dụng).

**Cấu hình Redis (nếu dùng cache):**
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` — xem `REDIS_CACHE_GUIDE.md` để biết chi tiết.

Xem file `.env.example` để biết đầy đủ các biến và mặc định.



### 3. Lấy API Keys

#### Serper API Key
1. Đăng ký tài khoản tại https://serper.dev/
2. Vào Dashboard và copy API key
3. Paste vào `SERPER_API_KEY` trong file `.env`

#### Gemini API Key (Tùy chọn - chỉ cần nếu dùng Gemini cho judging)
1. Truy cập https://aistudio.google.com/app/apikey
2. Tạo API key mới
3. Paste vào `GEMINI_API_KEY` trong file `.env`
4. Gán `FACTCHECKER_JUDGE_PROVIDER=gemini` trong file `.env`



## Triển Khai Nhanh

### Bước 1: Clone Repository

```bash
git clone https://github.com/idirlab/ClaimCheck.git
cd ClaimCheck
```

### Bước 2: Tạo Thư Mục Reports

```bash
mkdir -p reports
chmod 755 reports
```

### Bước 3: Tạo File `.env`

Tạo và điền file `.env` như hướng dẫn ở phần [Chuẩn Bị Biến Môi Trường](#chuẩn-bị-biến-môi-trường).

### Bước 4: Triển Khai

#### Triển Khai Với CPU (Mặc Định)

```bash
cd demo
docker compose -f docker-compose.yml up -d --build
```

#### Triển Khai Với GPU (Nếu Có NVIDIA GPU)

```bash
cd demo
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

**Giải thích lệnh:**
- `-f docker-compose.yml`: File cấu hình chính
- `-f docker-compose.gpu.yml`: File override để bật GPU
- `-d`: Chạy ở chế độ detached (background)
- `--build`: Build lại images nếu có thay đổi

### Bước 5: Kiểm Tra Trạng Thái

```bash
# Xem trạng thái các containers
docker compose -f docker-compose.yml ps

# Xem logs
docker compose -f docker-compose.yml logs -f
```

### Bước 6: Kiểm Tra Ứng Dụng

1. **Frontend**: Mở trình duyệt và truy cập `http://localhost`
2. **Backend API Docs**: Truy cập `http://localhost:8000/docs`
3. **Health Check**: 
   ```bash
   curl http://localhost:8000/health
   # Hoặc
   curl http://localhost/health
   ```

Kết quả mong đợi:
```json
{"message": "healthy"}
```

---

## Các lệnh thường dùng

### Dừng Ứng Dụng

```bash
cd demo
docker compose -f docker-compose.yml down
```

### Dừng và Xóa Tất Cả

```bash
cd demo
docker compose -f docker-compose.yml down -v
```

### Xem Logs

```bash
cd demo

# Xem logs của tất cả services
docker compose -f docker-compose.yml logs -f

# Xem logs của backend
docker compose -f docker-compose.yml logs -f backend

# Xem logs của frontend
docker compose -f docker-compose.yml logs -f frontend
```

### Rebuild Sau Khi Cập Nhật Code

```bash
cd demo

# Rebuild và restart
docker compose -f docker-compose.yml up -d --build

# Hoặc rebuild từ đầu (xóa cache)
docker compose -f docker-compose.yml build --no-cache
docker compose -f docker-compose.yml up -d
```

### Kiểm tra trạng thái container

```bash
# Xem trạng thái chi tiết
docker compose -f docker-compose.yml ps

# Test health endpoint
curl http://localhost:8000/health
```
