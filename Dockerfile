# Use an official Python image with CUDA support if needed
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables for non-interactive apt-get usage
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        git \
        curl \
        software-properties-common \
        build-essential \
        cmake \
        g++ \
        libopencv-dev \
        python3.10 \
        python3.10-distutils \
        python3.10-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip to avoid any compatibility issues
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /app/Hunyuan3D-2

# Copy all files to the container
COPY . /app/Hunyuan3D-2

# Install required Python packages
RUN pip install -r requirements.txt

# Build and install the custom rasterizer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py build_ext --inplace
RUN python3 setup.py install

# Build and install the differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py build_ext --inplace
RUN python3 setup.py install

# Expose the port for the API
EXPOSE 8080

# Set the default command to run the API server
WORKDIR /app/Hunyuan3D-2
CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
