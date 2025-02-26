# Use an official Python image with CUDA support if neededa
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variable for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app/Hunyuan3D-2

# Install system dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        software-properties-common \
        build-essential \
        cmake \
        g++ \
        libopencv-dev \
        wget \
        curl \
        git \
        tzdata \
        python3.10 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Configure timezone to avoid interactive prompt
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Manually install python3.10-distutils and python3.10-dev
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10-distutils python3.10-dev

# Make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip



# Copy all files to the container
COPY . /app/Hunyuan3D-2

# Install required Python packages
RUN pip install -r requirements.txt

# Build and install the custom rasterizer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# Build and install the differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# Expose the port for the API
EXPOSE 8080

# Set the default command to run the API server
WORKDIR /app/Hunyuan3D-2
CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
