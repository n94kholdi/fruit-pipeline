FROM python:3.11-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="fruit-pipeline" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    FRUIT_PIPELINE_DATA_DIR=/app/output/dashboard \
    FRUIT_PIPELINE_PALLET_CONFIG=/app/config/pallet_types.yaml \
    FRUIT_PIPELINE_DETECTOR_WEIGHTS=/models/yolo11x.pt \
    FRUIT_PIPELINE_SAM_CHECKPOINT=/models/sam_vit_l_0b3195.pth

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[api]"

COPY config ./config

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/output/dashboard \
    && chown -R appuser:appuser /app/output

USER appuser
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "fruit_pipeline.dashboard_api:app", "--host", "0.0.0.0", "--port", "8010"]
