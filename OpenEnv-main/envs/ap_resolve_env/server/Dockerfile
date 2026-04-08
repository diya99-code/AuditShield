FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire OpenEnv repo so we can install openenv-core from source
COPY . /app/repo

# Install openenv-core from source (includes all framework deps)
RUN pip install --no-cache-dir -e /app/repo

# Install environment-specific requirements
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    pydantic>=2.0.0 \
    uvicorn>=0.24.0 \
    websockets>=15.0.0 \
    httpx>=0.28.0 \
    gradio>=4.0.0

# Copy the ap_resolve_env into the working directory
COPY envs/ap_resolve_env /app/env

ENV PYTHONPATH="/app/env:/app/repo/src:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/env"]
