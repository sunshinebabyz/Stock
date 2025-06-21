FROM python:3.9-slim-buster

# Cài đặt các gói hệ thống cần thiết cho TensorFlow và các lib khác
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Cài các package trước, trừ tensorflow
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        flask pandas numpy plotly scipy scikit-learn statsmodels pmdarima yfinance joblib

# Cài tensorflow riêng từ source phù hợp
RUN pip install tensorflow==2.10.0 --extra-index-url https://google-coral.github.io/py-repo/

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
