FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/ddcomb

RUN python -m pip install -U pip setuptools wheel

COPY pyproject.toml README.md ./
COPY src ./src
COPY conf ./conf
COPY scripts ./scripts
COPY environment/requirements.txt ./environment/requirements.txt

# Install the project metadata only (no deps): torch/torchvision are provided by
# the base image and must not be touched.
RUN python -m pip install --no-deps "."

# Install the runtime dependencies (single source of truth). torch/torchvision
# are deliberately absent from requirements.txt for the same reason.
RUN python -m pip install -r environment/requirements.txt

# Smoke test: the full runtime surface must import cleanly.
RUN python - <<'PY'
import torch, torchvision
import hydra, omegaconf, torchmetrics, wandb
import timm, open_clip
import seaborn, umap, sklearn, yaml, PIL, pandas
import ddcomb
print("all imports OK")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
PY
