FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

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

# 6) Install huggingface_hub so we can do partial repo downloads
RUN pip install huggingface_hub

# 7) Create the local directory for the model subfolder
RUN mkdir -p /root/.cache/hy3dgen/tencent/Hunyuan3D-2/

###############################################################################
#     *** Download only the hunyuan3d-dit-v2-0 subfolder using allow_patterns ***
###############################################################################
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='tencent/Hunyuan3D-2', \
    local_dir='/root/.cache/hy3dgen/tencent/Hunyuan3D-2', \
    revision='main', \
    allow_patterns=['hunyuan3d-dit-v2-0/*'] \
)"


RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='tencent/Hunyuan3D-2', \
    local_dir='/root/.cache/hy3dgen/tencent/Hunyuan3D-2', \
    revision='main', \
    allow_patterns=['hunyuan3d-paint-v2-0/*'] \
)"



# Now your Docker image has only that specific folder from HF
RUN ls -l /root/.cache/hy3dgen/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0

# 8) (Optional) Install PyTorch (Nightly, CUDA 11.8)
RUN pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118

# Install SentencePiece for T5Tokenizer
RUN pip install sentencepiece

# 9) Copy your code
COPY . /app/Hunyuan3D-2

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"


# 10) Install other Python dependencies
RUN pip install --no-cache-dir -r /app/Hunyuan3D-2/requirements.txt

# 11) Build your custom rasterizer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# 12) Build your differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# 13) Expose port 8080 (if your API runs on 8080)
EXPOSE 8080

# 14) Default command to run your server
WORKDIR /app/Hunyuan3D-2

CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]


