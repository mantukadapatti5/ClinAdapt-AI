FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --retries 10 --timeout 120 --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 "numpy<2"
RUN pip install --retries 10 --timeout 120 -r requirements.txt

COPY app ./app
COPY verify_datasets.py .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
