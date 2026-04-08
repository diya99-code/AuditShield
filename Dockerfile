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
# Copy the full project structure
COPY . /app

# /app is on PYTHONPATH so `envs.ap_resolve_env` resolves correctly
ENV PYTHONPATH="/app:/app/openenv_src/src"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Expose the port (Hugging Face expects 7860 for Docker Spaces)
EXPOSE 7860

# Command to run the application using the new server.app module
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
