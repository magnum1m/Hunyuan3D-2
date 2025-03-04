# Stage 1: Build Stage
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app/Hunyuan3D-2

# 1) Install system dependencies + Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        cmake \
        g++ \
        wget \
        curl \
        git \
        tzdata \
        python3.10 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 2) Configure timezone
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# 3) Add deadsnakes + install Python 3.10 distutils/dev
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10-distutils python3.10-dev

# 4) Make Python 3.10 default + link pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 5) Upgrade pip from PyPI (bypass system pip)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 6) Install necessary Python packages
RUN pip install --no-cache-dir huggingface_hub runpod sentencepiece

# 7) Create cache directories for models
RUN mkdir -p /root/.cache/hy3dgen/tencent/Hunyuan3D-2/

# 8) Download model files during build
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='tencent/Hunyuan3D-2', \
    local_dir='/root/.cache/hy3dgen/tencent/Hunyuan3D-2', \
    revision='main', \
    allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-paint-v2-0/*'] \
)"

# 9) List downloaded files to verify
RUN ls -l /root/.cache/hy3dgen/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0

# 10) Install PyTorch (Nightly, CUDA 11.8) and dependencies
RUN pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118

# 11) Copy your code
COPY . /app/Hunyuan3D-2

# 12) Install other Python dependencies
RUN pip install --no-cache-dir -r /app/Hunyuan3D-2/requirements.txt

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
# 13) Build your custom rasterizer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# 14) Build your differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

COPY dummy_run.py /app/Hunyuan3D-2/
RUN python3 /app/Hunyuan3D-2/dummy_run.py

# 16) Environment variables for cach

ENV TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
ENV HF_HOME="/root/.cache/huggingface"
ENV OMP_NUM_THREADS=1

# 17) Enable FP16 precision for inference
RUN pip install --no-cache-dir nvidia-apex
RUN python3 -c "from apex import amp"  # Ensure Apex is installed

# 18) Expose port 8080 (if your API runs on 8080)
EXPOSE 8080

# 19) Default command to run your server
WORKDIR /app/Hunyuan3D-2
CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
