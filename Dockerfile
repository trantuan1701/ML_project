FROM python:3.11-slim

# Cài gói hệ thống tối thiểu (nếu bạn cần build xgboost/lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project vào container
COPY . .

# Render sẽ đặt biến môi trường PORT, ta dùng lại nó
ENV PORT=10000
EXPOSE 10000

# Chạy FastAPI
CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port ${PORT}"]
