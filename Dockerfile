FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    DATA_DIR=/app/data

WORKDIR /app

RUN groupadd --system --gid 10001 omnirag \
    && useradd --system --uid 10001 --gid omnirag --create-home omnirag

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY tests ./tests
RUN mkdir -p /app/data && chown omnirag:omnirag /app/data

EXPOSE 8000
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Root is used only to take ownership of the bind-mounted data directory;
# the server itself runs as the unprivileged omnirag user.
ENTRYPOINT ["/bin/sh", "-c", "if [ \"$(id -u)\" = 0 ]; then chown -R omnirag:omnirag /app/data && exec setpriv --reuid=omnirag --regid=omnirag --init-groups env HOME=/home/omnirag \"$@\"; fi; exec \"$@\"", "entrypoint"]

# Embedded Qdrant locks its storage, so exactly one worker is required.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
