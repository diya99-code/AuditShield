FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install openenv-core from source
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
# Structure: /app/envs/ap_resolve_env/ so `envs.ap_resolve_env` is importable
COPY envs /app/envs

# /app is on PYTHONPATH so `envs.ap_resolve_env` resolves correctly
ENV PYTHONPATH="/app:/app/openenv_src/src:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run from /app so relative imports in the package work
CMD ["python", "-m", "uvicorn", "envs.ap_resolve_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
