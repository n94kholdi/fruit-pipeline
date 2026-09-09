# fruit-pipeline base image
#
# Contains ONLY the Python base image plus the pinned PyTorch + torchvision
# CUDA build. This is the largest, slowest, and least frequently changing
# layer of the application image, so it lives in its own image that the
# application Dockerfile consumes via:
#
#   FROM ghcr.io/<owner>/fruit-pipeline-base:<tag>
#
# Rebuild and re-push this image ONLY when one of these changes:
#   * PyTorch version         (PYTORCH_VERSION)
#   * torchvision version     (TORCHVISION_VERSION)
#   * CUDA wheel flavor       (PYTORCH_CUDA_FLAVOR / the pinned index URL)
#   * Python base image       (the FROM line below)
#
# Do NOT add application dependencies (SAM 2, the project package, apt
# packages, etc.) here -- those stay in the application Dockerfile so normal
# application builds keep reusing this image unchanged.
FROM python:3.11-slim

ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG PYTORCH_CUDA_FLAVOR=cu118

# CUDA 11.8 runs on NVIDIA Linux drivers >= 450.80.02 and remains compatible
# with newer drivers. Override PYTORCH_CUDA_FLAVOR (cu121 or cu124 for these
# pinned PyTorch versions) when building for hardware that requires it, and
# publish the result under a matching tag.
RUN python -m pip install --no-cache-dir \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_FLAVOR}" \
    && python -c "import torch, torchvision; expected='${PYTORCH_CUDA_FLAVOR}'; actual='cu' + str(torch.version.cuda).replace('.', ''); assert torch.__version__.startswith('${PYTORCH_VERSION}+'), f'Unexpected PyTorch version: {torch.__version__}'; assert torchvision.__version__.startswith('${TORCHVISION_VERSION}+'), f'Unexpected torchvision version: {torchvision.__version__}'; assert actual == expected, f'Expected {expected}, installed {actual}'"

LABEL org.opencontainers.image.title="fruit-pipeline-base" \
      org.opencontainers.image.description="python:3.11-slim + pinned PyTorch/torchvision CUDA build for fruit-pipeline"
