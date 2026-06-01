FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# API defaults (override via docker run -e or --env-file .env)
ENV APP_PORT=3000
EXPOSE 3000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/')" || exit 1

ENTRYPOINT ["python", "-m", "src.main"]
