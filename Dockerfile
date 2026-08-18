FROM python:3.12-slim
WORKDIR /app_root

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Stamp the image with its build time. This is what makes a stale-image deploy
# visible: the value changes on every real rebuild and stays identical when
# --force-recreate silently reuses the old image, which .env changes do NOT,
# so code and config can otherwise disagree with nothing to show for it.
# Surfaced by GET /health as "build". Written after COPY so it is never served
# from a cached layer.
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /app_root/.build-stamp

# Git on Windows doesn't preserve the executable bit, so set it here rather
# than relying on the checkout — otherwise the container dies at startup with
# "permission denied" on the entrypoint.
RUN chmod +x scripts/resolve_public_url.sh

# Non-root for safety.
RUN useradd -m appuser && chown -R appuser /app_root
USER appuser

EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# Single worker: APScheduler runs in-process and must not start twice.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
