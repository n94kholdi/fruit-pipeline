# torch and torchvision are provided by this base image
# (python:3.11-slim + the pinned PyTorch/torchvision CUDA build).
# The base image is rebuilt only when Python, PyTorch, torchvision,
# or the CUDA wheel flavor changes.
FROM ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    FRUIT_PIPELINE_DATA_DIR=/app/output/dashboard \
    FRUIT_PIPELINE_PALLET_CONFIG=/app/config/pallet_types.yaml \
    FRUIT_PIPELINE_DETECTOR_WEIGHTS=/models/yolo11x.pt \
    FRUIT_PIPELINE_SAM_CHECKPOINT=/models/sam_vit_l_0b3195.pth \
    FRUIT_PIPELINE_SAM_USE_FP16=true \
    YOLO_CONFIG_DIR=/app/output/.config/Ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies in a separate layer.
# The temporary package is enough for pip to read dependency metadata and
# create console entry points. PYTHONPATH points those entry points at the
# real source copied below, so normal source edits do not invalidate the
# dependency layer.
#
# torch and torchvision are already installed by the reusable base image,
# so pip reuses them instead of downloading the CUDA wheels again.
COPY pyproject.toml README.md ./

RUN mkdir -p src/fruit_pipeline \
    && touch src/fruit_pipeline/__init__.py \
    && python -m pip install --no-cache-dir ".[api]" \
    && rm -rf src build fruit_pipeline.egg-info

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/output/dashboard /app/output/.config/Ultralytics \
    && chown -R appuser:appuser /app/output

COPY src ./src
COPY config ./config

USER appuser

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3)" || exit 1

# VERSION changes for every commit. Keep it after the expensive dependency
# and source layers so a new image label does not invalidate those cached
# layers.
ARG VERSION=dev

LABEL org.opencontainers.image.title="fruit-pipeline" \
      org.opencontainers.image.version="${VERSION}"

CMD ["python", "-m", "uvicorn", "fruit_pipeline.dashboard_api:app", "--host", "0.0.0.0", "--port", "8010"]