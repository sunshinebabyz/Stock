# Dockerfile - dùng cho Flask + TensorFlow + ML app trên Render

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
