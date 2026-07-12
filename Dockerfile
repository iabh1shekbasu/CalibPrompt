# CalibPrompt — reproducible environment
# Base image already provides the reference stack: PyTorch 2.1.0 + CUDA 12.1 + Python 3.10.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Build tools (Dassl's setup.py compiles a small numpy C extension) + git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/CalibPrompt
COPY . /workspace/CalibPrompt

# Python dependencies (numpy is pinned <2; nltk etc.), then the vendored Dassl.pytorch.
# --no-build-isolation lets Dassl's setup.py see the numpy we just installed.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r Dassl.pytorch/requirements.txt \
    && pip install --no-cache-dir -e Dassl.pytorch --no-build-isolation \
    && pip install --no-cache-dir "numpy==1.26.3"

# Datasets and Med-VLM weights are mounted at runtime — NOT baked into the image.
ENV DATASET_ROOT=/data \
    MODEL_ROOT=/models \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["/bin/bash"]
