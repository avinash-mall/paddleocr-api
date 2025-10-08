# Use NVIDIA CUDA runtime as base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PIP_DEFAULT_TIMEOUT=100
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    wget \
    curl \
    libx11-6 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Remove Windows-specific paddlepaddle-gpu line from requirements.txt
RUN sed -i '/paddlepaddle-gpu @ file/d' requirements.txt

# Upgrade pip to latest version
RUN python3 -m pip install --upgrade pip

# Install PaddlePaddle GPU 3.2.0 (required for PaddleX 3.2.1 compatibility)
# PaddleX 3.2.1 requires PaddlePaddle 3.x for set_optimization_level() method
# Ubuntu 22.04 has Python 3.10, so use cp310 wheel
RUN wget -q https://github.com/avinash-mall/paddleocr-api/releases/download/paddlepaddle_gpu_py310/paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl && \
    python3 -m pip install --no-cache-dir paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl && \
    rm paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl && \
    echo "✓ PaddlePaddle GPU 3.2.0 (cp310) installed successfully"

# Install minimal dependencies for the API
RUN python3 -m pip install --no-cache-dir -r requirements.txt uvicorn

# Copy the model download script
COPY download_models.py .

# Set HOME directory for PaddlePaddle model cache
ENV HOME=/root
ENV PADDLEOCR_HOME=/root/.paddleocr
ENV PADDLEX_HOME=/root/.paddlex

# Pre-download ALL language models with retry logic and validation
# Models are cached in /root/.paddlex/official_models/
# GPU errors during build are EXPECTED and handled correctly
RUN mkdir -p /root/.paddleocr /root/.paddlex && \
    python3 download_models.py && \
    echo "=== Model Cache Summary ===" && \
    echo "PaddleOCR cache: $(du -sh /root/.paddleocr 2>/dev/null || echo 'N/A')" && \
    echo "PaddleX cache: $(du -sh /root/.paddlex 2>/dev/null || echo 'N/A')" && \
    echo "Total files: $(find /root/.paddleocr /root/.paddlex -type f 2>/dev/null | wc -l)" && \
    echo "==========================="

# Create a default ocr_api.py (will be overridden by volume mount)
RUN echo 'print("OCR API placeholder - mount your ocr_api.py file")' > ocr_api.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000"]