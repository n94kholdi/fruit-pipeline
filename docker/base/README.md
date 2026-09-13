# fruit-pipeline base image

`docker/base/Dockerfile` builds `ghcr.io/n94kholdi/fruit-pipeline-base`, a
reusable image that contains **only**:

- the `python:3.11-slim` base image, and
- the pinned PyTorch + torchvision CUDA build
  (`torch==2.5.1`, `torchvision==0.20.1`, index URL
  `https://download.pytorch.org/whl/cu118`).

That PyTorch layer is the largest and slowest part of the application image and
almost never changes. Splitting it out means routine application builds pull a
ready-made layer from GHCR instead of downloading and installing ~2 GB of CUDA
wheels every time.

The application image (`../../Dockerfile`) consumes it with:

```dockerfile
FROM ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118
```

## Tag convention

The tag encodes everything the base image pins:

```
py<python>-torch<torch>-cu<cuda>
e.g. py3.11-torch2.5.1-cu118
```

Pick a new tag whenever any of those values change, and update the `FROM` line
in `../../Dockerfile` to match.

## When to rebuild

Rebuild and re-push the base image **only** when one of these changes:

| Change                       | Where it is defined                          |
| ---------------------------- | -------------------------------------------- |
| PyTorch version              | `ARG PYTORCH_VERSION` in `Dockerfile`        |
| torchvision version          | `ARG TORCHVISION_VERSION` in `Dockerfile`    |
| CUDA version / wheel flavor  | `ARG PYTORCH_CUDA_FLAVOR` + the index URL    |
| Python base image            | `FROM python:3.11-slim` in `Dockerfile`      |

**Everything else is a normal application build.** Adding, removing, or bumping
application dependencies (SAM 2, `pyproject.toml` extras, apt packages), editing
source, or cutting a release does **not** require touching this image — those
builds just reuse the published base image.

Keep these versions in sync with `pyproject.toml` and the CI test job in
`.github/workflows/build-image.yml` (which installs the same
`torch==2.5.1` / `torchvision==0.20.1` pair against the CPU wheel index).

## Build and push manually (first time and every later rebuild)

Requires Docker with Buildx and a GHCR login. Use a
[Personal Access Token](https://github.com/settings/tokens) with the
`write:packages` scope:

```bash
# 1. Authenticate to GHCR (once per machine)
echo "$GHCR_PAT" | docker login ghcr.io -u n94kholdi --password-stdin

# 2. Build the base image from the repo root
cd "$(git rev-parse --show-toplevel)"
docker build \
  -f docker/base/Dockerfile \
  -t ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118 \
  docker/base

# 3. Push it
docker push ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118
```

Optionally also move a `latest` tag:

```bash
docker tag \
  ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu118 \
  ghcr.io/n94kholdi/fruit-pipeline-base:latest
docker push ghcr.io/n94kholdi/fruit-pipeline-base:latest
```

### Building a different CUDA flavor

The pinned PyTorch release also ships `cu121` and `cu124` wheels. Build and push
them under matching tags without editing the Dockerfile:

```bash
docker build \
  -f docker/base/Dockerfile \
  --build-arg PYTORCH_CUDA_FLAVOR=cu121 \
  -t ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu121 \
  docker/base
docker push ghcr.io/n94kholdi/fruit-pipeline-base:py3.11-torch2.5.1-cu121
```

### Or use the manual GitHub Actions workflow

`.github/workflows/build-base-image.yml` does the same thing. It runs **only on
manual dispatch** (Actions tab → *Build fruit-pipeline base image* → *Run
workflow*), never automatically on push, so ordinary commits never rebuild the
base image. It takes the CUDA flavor and tag as inputs.

## Make the package pullable

The first push creates a private GHCR package. Either:

- mark `fruit-pipeline-base` **public** in the package settings (simplest — no
  credentials needed to pull), or
- keep it private and, on the package's *Manage Actions access* page, give the
  `fruit-pipeline` repository **Read** access so the application build's
  `GITHUB_TOKEN` can pull it.
