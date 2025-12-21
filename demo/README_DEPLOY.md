# Hướng Dẫn Triển Khai Hệ Thống ClaimCheck

Tài liệu này hướng dẫn chi tiết cách triển khai hệ thống ClaimCheck sử dụng Docker và Docker Compose.

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
3. [Chuẩn Bị](#chuẩn-bị)
4. [Triển Khai](#triển-khai)
5. [Kiểm Tra](#kiểm-tra)
6. [Troubleshooting](#troubleshooting)
7. [Cấu Trúc Hệ Thống](#cấu-trúc-hệ-thống)

---

## 🔧 Yêu Cầu Hệ Thống

### Phần Mềm Cần Thiết

- **Docker**: Phiên bản 20.10 trở lên
- **Docker Compose**: Phiên bản 2.0 trở lên
- **Git**: Để clone repository

### Kiểm Tra Cài Đặt

```bash
# Kiểm tra Docker
docker --version

# Kiểm tra Docker Compose
docker compose version

# Kiểm tra quyền truy cập Docker
docker ps
```

**Lưu ý**: Nếu gặp lỗi `permission denied`, thực hiện:

```bash
# Thêm user vào group docker
sudo usermod -aG docker $USER

# Áp dụng thay đổi (chọn một trong hai cách)
newgrp docker
# HOẶC
# Đăng xuất và đăng nhập lại
```

### Tài Nguyên Hệ Thống

- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: Tối thiểu 10GB trống
- **CPU**: Tối thiểu 2 cores

---

## 📁 Cấu Trúc Thư Mục

```
ClaimCheck/
├── demo/                    # Thư mục deployment
│   ├── app/                 # Backend code (FastAPI)
│   ├── src/                 # Frontend code (React)
│   ├── Dockerfile.backend   # Dockerfile cho backend
│   ├── Dockerfile.frontend  # Dockerfile cho frontend
│   ├── docker-compose.yml   # Cấu hình Docker Compose
│   ├── nginx.conf           # Cấu hình Nginx
│   ├── .env                 # Biến môi trường (tạo từ .env.example)
│   └── README_DEPLOY.md     # File này
├── factchecker/             # Module fact-checking
├── requirements.txt          # Python dependencies
└── reports/                 # Thư mục lưu báo cáo (tự động tạo)
```

---

## 🚀 Chuẩn Bị

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd ClaimCheck
```

### Bước 2: Tạo File .env

Tạo file `.env` trong thư mục `demo/`:

```bash
cd demo
cp .env.example .env  # Nếu có file .env.example
# HOẶC tạo file .env mới
nano .env
```

Nội dung file `.env`:

```env
# API Keys (BẮT BUỘC)
SERPER_API_KEY=your_serper_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_google_cx_here

# Cấu hình Fact-Checker (TÙY CHỌN)
FACTCHECKER_MODEL_NAME=qwen2.5:0.5b
FACTCHECKER_MAX_ACTIONS=2
```

**Lưu ý**: 
- Thay thế các giá trị `your_*_here` bằng API keys thực tế
- File `.env` chứa thông tin nhạy cảm, không commit vào Git

### Bước 3: Kiểm Tra Cấu Trúc

Đảm bảo các file sau tồn tại:

```bash
ls -la demo/
# Phải có:
# - Dockerfile.backend
# - Dockerfile.frontend
# - docker-compose.yml
# - nginx.conf
# - .env
```

---

## 🎯 Triển Khai

### Bước 1: Di Chuyển Vào Thư Mục Demo

```bash
cd demo
```

### Bước 2: Build Images

```bash
# Build tất cả services
docker compose build

# HOẶC build từng service riêng
docker compose build backend
docker compose build frontend
```

**Lưu ý**: 
- Lần đầu build có thể mất 10-15 phút (tải dependencies)
- Các lần build sau sẽ nhanh hơn nhờ Docker cache

### Bước 3: Khởi Động Services

```bash
# Khởi động tất cả services
docker compose up -d

# Xem logs
docker compose logs -f

# Xem logs của một service cụ thể
docker compose logs -f backend
docker compose logs -f frontend
```

### Bước 4: Kiểm Tra Trạng Thái

```bash
# Xem trạng thái containers
docker compose ps

# Kết quả mong đợi:
# NAME                  STATUS
# claimcheck-backend    Up (healthy)
# claimcheck-frontend   Up
```

---

## ✅ Kiểm Tra

### 1. Kiểm Tra Backend

```bash
# Health check
curl http://localhost:8000/health

# API Documentation
# Mở trình duyệt: http://localhost:8000/docs
```

### 2. Kiểm Tra Frontend

```bash
# Truy cập ứng dụng
# Mở trình duyệt: http://localhost
```

### 3. Kiểm Tra API Qua Nginx Proxy

```bash
# Health check qua proxy
curl http://localhost/api/health

# API Documentation qua proxy
# Mở trình duyệt: http://localhost/docs
```

### 4. Test Fact-Checking

1. Mở trình duyệt: `http://localhost`
2. Nhập một claim cần kiểm tra
3. Chọn ngày cắt (cut-off date)
4. Click "Submit"
5. Đợi kết quả (có thể mất 1-5 phút)

---

## 🔍 Troubleshooting

### Lỗi 1: Permission Denied

**Triệu chứng**:
```
permission denied while trying to connect to the Docker daemon socket
```

**Giải pháp**:
```bash
sudo usermod -aG docker $USER
newgrp docker
# HOẶC đăng xuất và đăng nhập lại
```

### Lỗi 2: Backend Không Khởi Động

**Triệu chứng**:
- Container `claimcheck-backend` có status `Restarting` hoặc `Exited`
- Logs hiển thị lỗi import

**Kiểm tra**:
```bash
# Xem logs chi tiết
docker compose logs backend --tail 100

# Kiểm tra container có đang chạy
docker compose ps
```

**Giải pháp**:
- Kiểm tra file `.env` có đầy đủ API keys
- Kiểm tra `requirements.txt` có đầy đủ dependencies
- Rebuild backend: `docker compose build backend`

### Lỗi 3: 502 Bad Gateway

**Triệu chứng**:
- Frontend hiển thị 502 Bad Gateway
- API calls thất bại

**Kiểm tra**:
```bash
# Kiểm tra backend có đang chạy
docker compose ps backend

# Test kết nối từ frontend container đến backend
docker compose exec frontend sh
wget -O- http://backend:8000/health
exit
```

**Giải pháp**:
- Đảm bảo backend đang chạy: `docker compose up -d backend`
- Kiểm tra logs backend: `docker compose logs backend`
- Rebuild backend nếu cần: `docker compose build backend && docker compose up -d backend`

### Lỗi 4: Import Error - libGL.so.1

**Triệu chứng**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**Giải pháp**:
- Đảm bảo `Dockerfile.backend` có cài đặt:
  ```dockerfile
  RUN apt-get update && apt-get install -y \
      libgl1 \
      libglx0 \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*
  ```
- Rebuild backend: `docker compose build backend`

### Lỗi 5: Frontend Không Kết Nối Được Backend

**Triệu chứng**:
- Frontend hiển thị "Network Error"
- Console hiển thị `ERR_EMPTY_RESPONSE`

**Kiểm tra**:
```bash
# Kiểm tra frontend có đang dùng đúng API URL
# Mở DevTools (F12) > Console
# Xem log: "API: baseURL: ..."
```

**Giải pháp**:
- Đảm bảo frontend build với `VITE_API_URL=/api` (hoặc không set, sẽ dùng `/api` mặc định)
- Rebuild frontend: `docker compose build frontend && docker compose up -d frontend`

### Lỗi 6: Models Không Tải Được

**Triệu chứng**:
- Backend khởi động rất lâu (>5 phút)
- Logs hiển thị đang download models

**Giải pháp**:
- Đây là hành vi bình thường lần đầu chạy
- Models sẽ được cache, các lần sau sẽ nhanh hơn
- Nếu quá lâu, kiểm tra kết nối internet

### Lệnh Hữu Ích

```bash
# Xem logs real-time
docker compose logs -f

# Restart một service
docker compose restart backend
docker compose restart frontend

# Stop tất cả services
docker compose down

# Stop và xóa volumes (CẨN THẬN: mất dữ liệu)
docker compose down -v

# Rebuild và restart
docker compose build && docker compose up -d

# Xem resource usage
docker stats

# Vào trong container để debug
docker compose exec backend bash
docker compose exec frontend sh
```

---

## 🏗️ Cấu Trúc Hệ Thống

### Kiến Trúc

```
┌─────────────────┐
│   Browser       │
│  (User)         │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Nginx         │  Port 80
│  (Frontend)     │
└────────┬────────┘
         │
         ├─── / ────────────► Serve React App
         │
         └─── /api ─────────► Proxy to Backend
                              │
                              ▼
                    ┌─────────────────┐
                    │   FastAPI       │  Port 8000
                    │  (Backend)      │
                    └────────┬────────┘
                             │
                             ├─── FactChecker Module
                             ├─── Ollama (LLM)
                             └─── Reports Storage
```

### Ports

- **Port 80**: Frontend (Nginx) - Truy cập ứng dụng
- **Port 8000**: Backend (FastAPI) - API trực tiếp (nếu cần)

### Volumes

- `../reports:/app/../reports`: Lưu trữ báo cáo fact-checking

### Networks

- Docker Compose tự động tạo network `demo_default`
- Backend và Frontend giao tiếp qua tên service: `backend:8000`

---

## 📝 Các Lệnh Quản Lý

### Khởi Động

```bash
# Khởi động tất cả services
docker compose up -d

# Khởi động một service cụ thể
docker compose up -d backend
docker compose up -d frontend
```

### Dừng

```bash
# Dừng tất cả services
docker compose down

# Dừng một service cụ thể
docker compose stop backend
docker compose stop frontend
```

### Restart

```bash
# Restart tất cả services
docker compose restart

# Restart một service cụ thể
docker compose restart backend
```

### Rebuild

```bash
# Rebuild tất cả services
docker compose build

# Rebuild một service cụ thể
docker compose build backend
docker compose build frontend

# Rebuild và restart
docker compose build backend && docker compose up -d backend
```

### Xem Logs

```bash
# Xem logs tất cả services
docker compose logs -f

# Xem logs một service
docker compose logs -f backend
docker compose logs -f frontend

# Xem logs với số dòng giới hạn
docker compose logs --tail 100 backend
```

### Kiểm Tra Trạng Thái

```bash
# Xem trạng thái containers
docker compose ps

# Xem resource usage
docker stats

# Xem processes trong container
docker top claimcheck-backend
docker top claimcheck-frontend
```

---

## 🔐 Bảo Mật

### File .env

- **KHÔNG** commit file `.env` vào Git
- Đảm bảo file `.env` có quyền đọc phù hợp: `chmod 600 .env`
- Sử dụng `.env.example` làm template (không chứa giá trị thực)

### API Keys

- Bảo mật API keys, không chia sẻ công khai
- Rotate API keys định kỳ
- Sử dụng environment variables thay vì hardcode

### Network

- Backend chỉ expose port 8000 trên localhost (không public)
- Frontend expose port 80 (có thể cần firewall nếu public)

---

## 📊 Monitoring

### Health Checks

Backend có health check tự động:

```bash
# Kiểm tra health status
curl http://localhost:8000/health

# Xem health status trong docker compose
docker compose ps
```

### Logs

Logs được lưu tự động và có thể xem qua:

```bash
docker compose logs -f
```

### Reports

Báo cáo fact-checking được lưu trong thư mục `../reports/` (tương đối với thư mục `demo/`)

---

## 🚀 Production Deployment

### Khuyến Nghị

1. **Sử dụng Reverse Proxy**: Đặt Nginx hoặc Traefik phía trước
2. **SSL/TLS**: Cài đặt HTTPS với Let's Encrypt
3. **Monitoring**: Sử dụng Prometheus, Grafana
4. **Backup**: Backup thư mục `reports/` định kỳ
5. **Resource Limits**: Đặt limits cho containers trong `docker-compose.yml`

### Ví Dụ Resource Limits

Thêm vào `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:

1. Logs: `docker compose logs -f`
2. Status: `docker compose ps`
3. Health: `curl http://localhost:8000/health`
4. Documentation: Xem phần Troubleshooting ở trên

---

## 📄 License

[Thêm thông tin license nếu có]

---

**Chúc bạn triển khai thành công! 🎉**



