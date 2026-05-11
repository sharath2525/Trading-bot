FROM python:3.12-slim

WORKDIR /app

# Install dependencies directly (no poetry needed)
RUN pip install --no-cache-dir \
    hyperliquid-python-sdk \
    anthropic \
    python-dotenv \
    aiohttp \
    requests \
    web3 \
    rich

# Copy source
COPY src ./src

# API defaults
ENV APP_PORT=3000
EXPOSE 3000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/')" || exit 1

ENTRYPOINT ["python", "-m", "src.main"]
