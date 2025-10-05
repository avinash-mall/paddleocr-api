# Use NVIDIA CUDA runtime as base image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PIP_DEFAULT_TIMEOUT=100

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
    libgthread-2.0-0 \
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

# Copy the local paddlepaddle-gpu wheel (cp312 for Python 3.12)
COPY paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl /tmp/

# Remove Windows-specific paddlepaddle-gpu line
RUN sed -i '/paddlepaddle-gpu @ file/d' requirements.txt

# Try to download paddlepaddle-gpu wheel online first
RUN echo "Attempting to download PaddlePaddle-GPU 3.2.0 for cp312..." && \
    (python3 -c "import urllib.request; urllib.request.urlretrieve('https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl', 'paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl')" || \
     wget --timeout=30 --tries=3 https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl -O paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl || \
     curl --connect-timeout 30 --max-time 60 https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl -o paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl || \
     echo "Online download failed, will use local wheel") && \
    echo "Download attempt completed"

# Install paddlepaddle-gpu: try online download first, then local wheel
RUN if [ -f paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl ] && [ $(stat -c%s paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl 2>/dev/null || echo 0) -gt 100000000 ]; then \
        echo "✓ Installing PaddlePaddle-GPU 3.2.0 from downloaded cp312 wheel..." && \
        python3 -m pip install --break-system-packages --no-cache-dir paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl; \
    elif [ -f /tmp/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl ]; then \
        echo "✓ Installing PaddlePaddle-GPU 3.2.0 from local cp312 wheel..." && \
        python3 -m pip install --break-system-packages --no-cache-dir /tmp/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl; \
    else \
        echo "✗ ERROR: No valid PaddlePaddle-GPU 3.2.0 wheel found!" && \
        exit 1; \
    fi

# Install minimal dependencies for the API
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt uvicorn

# Copy the model download script
COPY download_models.py .

# Pre-download ALL language models with retry logic and validation
# This ensures all models are properly baked into the container
# 
# MODEL COVERAGE (ACTUAL SUPPORTED LANGUAGES):
# - PP-OCRv5: 5 languages (en, ch, japan, korean, chinese_cht) - Optimized for mixed-language documents
# - PP-OCRv3: 21 languages - Actually supported languages (not 80+ as claimed)
# - PP-StructureV3: Language-agnostic - Document structure analysis
# - PP-ChatOCRv4: Language-agnostic - Intelligent information extraction
#
# TOTAL STORAGE: ~3-5GB for actual model coverage (much smaller than claimed 17GB)
# BENEFITS: Zero runtime downloads, instant API responses, guaranteed availability
RUN python3 download_models.py

# Create a default ocr_api.py (will be overridden by volume mount)
RUN echo 'print("OCR API placeholder - mount your ocr_api.py file")' > ocr_api.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000"]