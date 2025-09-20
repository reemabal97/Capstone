FROM python:3.11-slim

# نخلي pip ياخذ wheels جاهزة ونضيف مستودع Torch CPU
ENV PIP_ONLY_BINARY=:all:
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# باكدجات نظام خفيفة (تفيد إن احتاجت wheel معين)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ثبّت المتطلبات
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# انسخ بقية المشروع
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
