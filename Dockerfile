# torch and torchvision are provided by this base image (python:3.11-slim +
# the pinned PyTorch/torchvision CUDA build). It is defined in
# docker/base/Dockerfile and rebuilt manually only when the PyTorch,
# torchvision, CUDA, or Python base image changes -- see docker/base/README.md.
FROM ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    FRUIT_PIPELINE_DATA_DIR=/app/output/dashboard \
    FRUIT_PIPELINE_PALLET_CONFIG=/app/config/pallet_types.yaml \
    FRUIT_PIPELINE_DETECTOR_WEIGHTS=/models/yolo11x.pt \
    FRUIT_PIPELINE_INFERENCE_MODE=detector \
    # FRUIT_PIPELINE_SAM2_MODEL=sam2.1_hiera_base_plus \
    FRUIT_PIPELINE_SAM2_MODEL=sam2.1_hiera_large \
    SAM2_PRECISION=bf16 \
    SAM2_RUNTIME=pytorch \
    SAM2_VOS_OPTIMIZED=false \
    SAM2_BUILD_CUDA=0 \
    YOLO_CONFIG_DIR=/app/output/.config/Ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install SAM 2 from Git in its own layer. --no-build-isolation makes pip build
# it against the PyTorch from the base image instead of provisioning a throwaway
# build env and pulling a second, PyPI/CUDA-12 copy of torch (plus its nvidia-*
# wheels) into this layer -- the ~3 GB duplicate that bloated the image. The
# slim base has no nvcc, so SAM2_BUILD_CUDA=0 skips the CUDA extension cleanly
# (runtime uses SAM2_VOS_OPTIMIZED=false and does not need it).
ARG SAM2_REF=c2ec8e14a185632b0a5d8b161928ceb50197eddc
RUN python -m pip install --no-cache-dir --no-build-isolation \
        "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}"

# Install the remaining application dependencies in a separate layer. The
# temporary package is enough for pip to read dependency metadata and create the
# console entry points. PYTHONPATH points those entry points at the real source
# copied below, so normal source edits do not invalidate either dependency layer.
# SAM 2 is already satisfied by the pinned commit above; --no-build-isolation
# keeps pip from re-provisioning a torch-carrying build env if it re-checks it.
COPY pyproject.toml README.md ./
RUN mkdir -p src/fruit_pipeline \
    && touch src/fruit_pipeline/__init__.py \
    && python -m pip install --no-cache-dir --no-build-isolation ".[api]" \
    && rm -rf src build fruit_pipeline.egg-info SAM-2.egg-info

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/output/dashboard /app/output/.config/Ultralytics \
    && chown -R appuser:appuser /app/output

COPY src ./src
COPY config ./config

USER appuser
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3)" || exit 1

# VERSION changes for every commit. Keep it after the expensive dependency and
# source layers so a new image label does not invalidate those cached layers.
ARG VERSION=dev
LABEL org.opencontainers.image.title="fruit-pipeline" \
      org.opencontainers.image.version="${VERSION}"

CMD ["python", "-m", "uvicorn", "fruit_pipeline.dashboard_api:app", "--host", "0.0.0.0", "--port", "8010"]
