# Stage 1: Build Stage
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app/Hunyuan3D-2

# 1) Install system dependencies + Python 3.10 + deadsnakes + Python distutils/dev
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
        python3-pip \
        python3.10-distutils \
        python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Configure timezone
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# 3) Make Python 3.10 default + link pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 4) Upgrade pip from PyPI (bypass system pip)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 5) Install PyTorch (CUDA 12.2) and dependencies + other Python packages
RUN pip install --no-cache-dir torch torchvision torchaudio huggingface_hub runpod sentencepiece --extra-index-url https://download.pytorch.org/whl/cu122

# 6) Install Apex for mixed precision training
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    pip install --no-cache-dir -v --disable-pip-version-check --global-option="--cpp_ext" --global-option="--cuda_ext" .

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

# 10) Install other Python dependencies using requirements.txt
COPY requirements.txt /app/Hunyuan3D-2/requirements.txt
RUN pip install --no-cache-dir -r /app/Hunyuan3D-2/requirements.txt

# 11) Copy source code
COPY . /app/Hunyuan3D-2

# 12) Build custom rasterizer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# 13) Build differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# 14) Clean up unnecessary files to reduce image size
RUN rm -rf /app/Hunyuan3D-2/.git

# 15) Environment variables for cache and performance
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
ENV HF_HOME="/root/.cache/huggingface"
ENV OMP_NUM_THREADS=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"  # Optimized for modern GPUs

# 16) Expose port 8080 for the API
EXPOSE 8080

# Stage 2: Create a smaller final image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

WORKDIR /app/Hunyuan3D-2

# Copy only necessary files from the builder stage
COPY --from=builder /app/Hunyuan3D-2 /app/Hunyuan3D-2
COPY --from=builder /root/.cache/hy3dgen /root/.cache/hy3dgen
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables for runtime
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
ENV HF_HOME="/root/.cache/huggingface"
ENV OMP_NUM_THREADS=1

# Default command to run your server
CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
