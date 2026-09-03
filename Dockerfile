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
    FRUIT_PIPELINE_SAM_CHECKPOINT=/models/sam_vit_l_0b3195.pth \
    YOLO_CONFIG_DIR=/app/output/.config/Ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install the project dependencies before copying application source.  The
# temporary package is enough for pip to read the dependency metadata and
# create the console entry points.  PYTHONPATH points those entry points at
# the real source copied below.  Consequently, normal src/ edits do not
# invalidate the expensive PyTorch/OpenCV/Ultralytics installation layer.
COPY pyproject.toml README.md ./
ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG PYTORCH_CUDA_FLAVOR=cu118

# CUDA 11.8 runs on NVIDIA Linux drivers >= 450.80.02 and remains compatible
# with newer drivers. Override PYTORCH_CUDA_FLAVOR (cu121 or cu124 for these
# pinned PyTorch versions) when deploying to hardware that requires it.
RUN python -m pip install --no-cache-dir \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_FLAVOR}" \
    && mkdir -p src/fruit_pipeline \
    && touch src/fruit_pipeline/__init__.py \
    && python -m pip install --no-cache-dir ".[api]" \
    && python -c "import torch; expected='${PYTORCH_CUDA_FLAVOR}'; actual='cu' + str(torch.version.cuda).replace('.', ''); assert torch.__version__.startswith('${PYTORCH_VERSION}+'), f'Unexpected PyTorch version: {torch.__version__}'; assert actual == expected, f'Expected {expected}, installed {actual}'" \
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

CMD ["python", "-m", "uvicorn", "fruit_pipeline.dashboard_api:app", "--host", "0.0.0.0", "--port", "8010"]
