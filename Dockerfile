# Dockerfile
FROM python:3.10-slim

# Recommended env settings for smoother logs + faster installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the repo (app.py + artifacts/)
COPY . .

# Koyeb will route traffic to this port (we run uvicorn on 8000)
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
