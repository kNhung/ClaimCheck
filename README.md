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

## Chuẩn Bị Biến Môi Trường

### 1. Tạo File `.env`

Copy .env.example sang file `.env` trong thư mục gốc của project (`ClaimCheck/`):

```bash
cd /path/to/ClaimCheck
cp .env.example .env
```

### 2. Cấu Hình Các Biến Môi Trường

Sau khi copy file `.env.example` sang `.env`, mở file `.env` và điền các giá trị thực tế:

**Các biến bắt buộc:**
- `SERPER_API_KEY`: API key từ Serper (bắt buộc)

**Các biến tùy chọn (có giá trị mặc định):**
- `FACTCHECKER_MODEL_NAME`: Tên model Ollama chính (mặc định: `qwen2.5:0.5b`)
- `FACTCHECKER_JUDGE_MODEL`: Model Ollama cho judging (tùy chọn, ví dụ: `qwen2.5:3b`)
- `FACTCHECKER_JUDGE_PROVIDER`: Provider cho judging (`ollama` hoặc `gemini`, mặc định: `ollama`)
- `FACTCHECKER_EMBED_DEVICE`: Thiết bị chạy embedding (`cpu` hoặc `cuda`, mặc định: `cpu`). Nếu đặt `cuda` nhưng GPU không khả dụng, hệ thống sẽ tự động fallback về `cpu`.
- `FACTCHECKER_MAX_ACTIONS`: Số lượng actions tối đa (mặc định: `1`)

**Các biến cho Gemini (cần khi `FACTCHECKER_JUDGE_PROVIDER=gemini`):**
- `GEMINI_API_KEY`: API key từ Google Gemini (bắt buộc nếu dùng Gemini)
- `GEMINI_MODEL`: Tên model Gemini (mặc định: `gemini-1.5-flash`)

Xem file `.env.example` để biết format và các biến có sẵn.

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
