# Use an official Python image with CUDA support if needed
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the working directory
WORKDIR /app/Hunyuan3D-2

# Copy all files to the container
COPY . /app/Hunyuan3D-2

# Install required Python packages
RUN pip install --upgrade pip
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
