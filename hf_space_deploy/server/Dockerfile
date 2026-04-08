FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy openenv source and install it
COPY openenv_src /app/openenv_src
RUN pip install --no-cache-dir -e /app/openenv_src

# Install additional runtime dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    pydantic>=2.0.0 \
    uvicorn>=0.24.0 \
    websockets>=15.0.0 \
    httpx>=0.28.0 \
    gradio>=4.0.0

# Copy the environment code
COPY . /app/env

ENV PYTHONPATH="/app/env:/app/openenv_src/src:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/env"]
