# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG GEMMA_ATTN_SHA=c6a0cf860612792597382e7745d1514a52d6ca58
ARG SAE_TRAINER_SHA=319ba616e4562d98cd3603fd6551dfcf86393308

LABEL org.opencontainers.image.title="gemma-4-e2b-it-base" \
      org.opencontainers.image.description="CUDA 12.8 / PyTorch 2.9.1 development image for Gemma-4-E2B-it attention and SAE work; no model weights." \
      org.opencontainers.image.source="https://github.com/JuiceB0xC0de/gemma-4-e2b-it-cu128-torch291-py311" \
      org.opencontainers.image.licenses="MIT" \
      ai.model.id="google/gemma-4-E2B-it" \
      ai.gemma.attention.source="https://github.com/zzhhjjj/gemma-triton-flash-attn" \
      ai.gemma.attention.revision="${GEMMA_ATTN_SHA}" \
      ai.sae.trainer.source="https://github.com/JuiceB0xC0de/event-aware-SAE-trainer" \
      ai.sae.trainer.revision="${SAE_TRAINER_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        build-essential \
        cmake \
        pkg-config \
        git \
        curl \
        wget \
        rsync \
        unzip \
        tar \
        less \
        vim \
        nano \
        tmux \
        btop \
        htop \
        procps \
        psmisc \
        lsof \
        strace \
        gdb \
        jq \
        tini \
        libaio-dev \
        ninja-build \
        numactl \
        cuda-command-line-tools-12-8 \
        nsight-systems-2025.6.3 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1

COPY requirements.txt constraints-cu128.txt /tmp/
RUN python -m pip install -c /tmp/constraints-cu128.txt -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/constraints-cu128.txt

RUN mkdir -p /opt/src \
    && git init /opt/src/gemma-triton-flash-attn \
    && git -C /opt/src/gemma-triton-flash-attn remote add origin https://github.com/zzhhjjj/gemma-triton-flash-attn.git \
    && git -C /opt/src/gemma-triton-flash-attn fetch --depth 1 origin "${GEMMA_ATTN_SHA}" \
    && git -C /opt/src/gemma-triton-flash-attn checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/src/gemma-triton-flash-attn rev-parse HEAD)" = "${GEMMA_ATTN_SHA}" \
    && git init /opt/src/event-aware-SAE-trainer \
    && git -C /opt/src/event-aware-SAE-trainer remote add origin https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git \
    && git -C /opt/src/event-aware-SAE-trainer fetch --depth 1 origin "${SAE_TRAINER_SHA}" \
    && git -C /opt/src/event-aware-SAE-trainer checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/src/event-aware-SAE-trainer rev-parse HEAD)" = "${SAE_TRAINER_SHA}"

RUN python -m pip install --no-deps -e /opt/src/gemma-triton-flash-attn \
    && python -m pip install --no-deps -e /opt/src/event-aware-SAE-trainer

ENV GEMMA_ATTN_SHA="${GEMMA_ATTN_SHA}" \
    SAE_TRAINER_SHA="${SAE_TRAINER_SHA}" \
    PYTHONPATH=/opt/src/event-aware-SAE-trainer:/opt/src/gemma-triton-flash-attn \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_HOME=/workspace/.cache/torch \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    WANDB_DIR=/workspace/.wandb \
    TOKENIZERS_PARALLELISM=false

COPY scripts/ /opt/image-tools/
RUN chmod 0755 /opt/image-tools/*.py \
    && python /opt/image-tools/verify_image.py

RUN mkdir -p \
        /workspace/.cache/huggingface/hub \
        /workspace/.cache/torch \
        /workspace/.cache/triton \
        /workspace/.cache/pip \
        /workspace/.wandb

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]
