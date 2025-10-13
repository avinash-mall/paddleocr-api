# PaddleOCR API - Multi-Language Document Intelligence

A production-ready FastAPI service providing GPU-accelerated OCR and document understanding with support for 21 languages and pre-downloaded models.

## Features

- **🌍 21 Languages**: Comprehensive international language support (English, Arabic, Hindi, Chinese, French, German, Spanish, Italian, Russian, Japanese, Korean, Portuguese, Dutch, Polish, Ukrainian, Thai, Vietnamese, Indonesian, Tamil, Telugu, Traditional Chinese)
- **⚡ GPU Acceleration**: NVIDIA CUDA 12.4 support (~1-2 seconds per page)
- **📄 Multiple Pipelines**: PP-OCRv5, PP-OCRv3, PP-StructureV3, PP-ChatOCRv4
- **🔄 Pre-downloaded Models**: All models baked into container (~1GB) for instant availability
- **🎯 High Accuracy**: PP-OCRv5 with 13% improvement, PP-OCRv3 with 21 language support
- **🔍 Zero Runtime Downloads**: No model download delays during API calls
- **🌐 Document Parsing**: Convert documents to Markdown or extract structured JSON
- **🤖 Intelligent Extraction**: LLM-powered information extraction with Ollama integration

## Prerequisites

- **Hardware**: NVIDIA GPU with CUDA support
- **Software**:
  - Docker with NVIDIA Container Toolkit
  - Docker Compose
  - NVIDIA GPU drivers (compatible with CUDA 12.4)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t paddleocr-api:latest .
```

Build time: 15-30 minutes (includes downloading and validating all 21 language models)

### 2. Start the Service

```bash
docker compose up -d
```

The API will be available at `http://localhost:8001`

### 3. Test the API

```bash
# Test English OCR (PP-OCRv5 - fastest for English)
curl -X POST http://localhost:8001/ocr/ppocrv5 \
  -F "file=@document.jpg" \
  -F "lang=en"

# Test Arabic OCR (PP-OCRv3 - supports 21 languages)
curl -X POST http://localhost:8001/ocr/ppocrv3 \
  -F "file=@arabic_doc.jpg" \
  -F "lang=ar"

# Convert document to Markdown
curl -X POST http://localhost:8001/ocr/structurev3/markdown \
  -F "file=@document.pdf" \
  -F "lang=en"
```

### 4. Access Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Language Reference**: http://localhost:8001/languages

## API Endpoints

### 1. PP-OCRv5 - Universal Text Recognition
**Endpoint**: `POST /ocr/ppocrv5`

Extract text from images and PDFs with 13% accuracy improvement over previous versions.

**Supported Languages** (5): English, Chinese, Japanese, Korean, Traditional Chinese

**Parameters**:
- `file`: Image (JPG, PNG) or PDF file
- `lang`: Language code (`en`, `ch`, `japan`, `korean`, `chinese_cht`)

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/ppocrv5 \
  -F "file=@document.jpg" \
  -F "lang=en"
```

### 2. PP-OCRv3 - Multi-Language Text Recognition
**Endpoint**: `POST /ocr/ppocrv3`

Comprehensive international language coverage with 21 supported languages.

**Supported Languages** (21): English, Arabic, Hindi, Chinese, French, German, Spanish, Italian, Russian, Japanese, Korean, Portuguese, Dutch, Polish, Ukrainian, Thai, Vietnamese, Indonesian, Tamil, Telugu, Traditional Chinese

**Parameters**:
- `file`: Image (JPG, PNG) or PDF file
- `lang`: Language code (e.g., `en`, `ar`, `hi`, `ch`, `fr`, `de`, `es`)

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/ppocrv3 \
  -F "file=@document.jpg" \
  -F "lang=ar"
```

### 3. PP-StructureV3 Markdown - Document to Markdown
**Endpoint**: `POST /ocr/structurev3/markdown`

Convert complex documents to clean Markdown format with preserved structure.

**Parameters**:
- `file`: Document image or PDF file
- `lang`: Language code for OCR recognition

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/structurev3/markdown \
  -F "file=@document.pdf" \
  -F "lang=en"
```

### 4. PP-StructureV3 JSON - Structured Document Data
**Endpoint**: `POST /ocr/structurev3/json`

Extract layout blocks, regions, and text elements with detailed bounding boxes.

**Parameters**:
- `file`: Document image or PDF file
- `lang`: Language code for OCR recognition

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/structurev3/json \
  -F "file=@document.pdf" \
  -F "lang=en"
```

### 5. PP-ChatOCRv4 - Intelligent Information Extraction
**Endpoint**: `POST /ocr/chatocrv4`

Extract specific fields using Ollama LLM integration for intelligent key-value extraction.

**Parameters**:
- `file`: Document image or PDF file
- `keys`: Comma-separated list of fields to extract (e.g., "Invoice Number,Date,Total")
- `lang`: Language code for OCR recognition
- `mllm_model`: Multimodal model (default: `llava:latest`)
- `llm_model`: Text LLM for extraction (default: `llama3:latest`)
- `ollama_base_url`: Ollama base URL (default: `http://localhost:11434`)

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/chatocrv4 \
  -F "file=@invoice.pdf" \
  -F "keys=Invoice Number,Date,Total" \
  -F "lang=en"
```

**Note**: Requires Ollama running locally. Install from https://ollama.ai

### 6. Language Reference
**Endpoint**: `GET /languages`

Get complete list of all supported languages with codes and usage guidance.

**Example**:
```bash
curl http://localhost:8001/languages
```

## Supported Languages

### PP-OCRv5 (5 Languages)
- `en` - English
- `ch` - Chinese (Simplified)
- `japan` - Japanese
- `korean` - Korean
- `chinese_cht` - Chinese (Traditional)

### PP-OCRv3 (21 Languages)
| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ar` | Arabic |
| `ch` | Chinese (Simplified) | `hi` | Hindi |
| `fr` | French | `de` | German |
| `es` | Spanish | `it` | Italian |
| `ru` | Russian | `japan` | Japanese |
| `korean` | Korean | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish |
| `uk` | Ukrainian | `th` | Thai |
| `vi` | Vietnamese | `id` | Indonesian |
| `ta` | Tamil | `te` | Telugu |
| `chinese_cht` | Traditional Chinese | | |

### Mixed Language Documents
- **Chinese + English**: Use `lang=ch` (PP-OCRv5 or PP-OCRv3)
- **Japanese + English**: Use `lang=japan` (PP-OCRv5 or PP-OCRv3)
- **Korean + English**: Use `lang=korean` (PP-OCRv5 or PP-OCRv3)
- **Arabic + English**: Use `lang=ar` (PP-OCRv3)
- **Other combinations**: Use primary language code

## Configuration

### Environment Variables

Configure via `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0           # GPU device ID
  - PADDLE_DEVICE=gpu:0              # PaddlePaddle device
  - PADDLEOCR_HOME=/root/.paddleocr  # Model cache directory
  - PADDLEX_HOME=/root/.paddlex      # PaddleX cache directory
```

### Pipeline Feature Flags

All endpoints support advanced preprocessing options:

- `use_doc_preprocessor`: Enable document preprocessing (default: `false`)
- `use_doc_orientation_classify`: Auto-detect page orientation (default: `true`)
- `use_doc_unwarping`: Perspective/unwarping correction (default: `false`)
- `use_textline_orientation`: Correct rotated text lines (default: `false`)

**Example**:
```bash
curl -X POST http://localhost:8001/ocr/ppocrv5 \
  -F "file=@document.jpg" \
  -F "lang=en" \
  -F "use_doc_orientation_classify=true" \
  -F "use_doc_unwarping=true"
```

## Performance

### Model Sizes
- PP-OCRv5: ~50MB per language (5 languages = 250MB)
- PP-OCRv3: ~30MB per language (21 languages = 630MB)
- PP-StructureV3: ~100MB (language-agnostic)
- PP-ChatOCRv4: ~80MB (language-agnostic)
- **Total**: ~1GB for complete model coverage

### Response Times
- **First request**: 1-2 seconds per page (models cached)
- **Subsequent requests**: 1-2 seconds per page
- **GPU utilization**: 18-21% during inference

### Accuracy Improvements
- PP-OCRv5: 13% improvement over PP-OCRv4
- PP-OCRv3: Optimized for 21 languages
- PP-StructureV3: Outperforms commercial solutions
- PP-ChatOCRv4: 15% improvement with ERNIE 4.5

## Docker Compose Configuration

The provided `docker-compose.yml` includes:
- **Port Mapping**: `8001:8000` (host:container)
- **GPU Support**: NVIDIA runtime with GPU reservation
- **Volume Mount**: Mount `ocr_api.py` for development
- **Health Check**: Automatic service health monitoring
- **Auto-restart**: Service restarts unless manually stopped

### Custom Port

To use a different port, edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Use port 8080 instead
```

## Development

### Local Development (without Docker)

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install paddlepaddle-gpu
pip install uvicorn
```

2. Pre-download models:
```bash
python download_models.py
```

3. Run the API:
```bash
uvicorn ocr_api:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

Access the interactive API documentation at `http://localhost:8001/docs` to test all endpoints with a user-friendly interface.

## Troubleshooting

### GPU Not Available
If you see "GPU requested but CUDA not available":
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
3. Ensure NVIDIA Container Toolkit is installed

### Model Download Failures
Models are pre-downloaded during Docker build. If you encounter issues:
1. Increase build timeout: `docker build --timeout 3600 -t paddleocr-api:latest .`
2. Check internet connectivity during build
3. Review build logs for specific errors

### Ollama Integration (ChatOCRv4)
To use PP-ChatOCRv4:
1. Install Ollama: https://ollama.ai
2. Pull required models:
   ```bash
   ollama pull llava:latest
   ollama pull llama3:latest
   ```
3. Ensure Ollama is accessible from Docker container

### Memory Issues
If you encounter out-of-memory errors:
1. Reduce concurrent requests
2. Process smaller images/documents
3. Increase GPU memory allocation
4. Use CPU mode: Set `PADDLE_DEVICE=cpu` in `docker-compose.yml`

## Project Structure

```
paddleocr/
├── ocr_api.py           # FastAPI application
├── download_models.py   # Model download script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
└── README.md          # This file
```

## Dependencies

- **FastAPI**: Modern web framework for APIs
- **PaddleOCR**: Open-source OCR toolkit
- **PaddleX**: Extended PaddlePaddle models
- **PaddlePaddle GPU**: Deep learning framework with CUDA support
- **Pillow**: Image processing library
- **NumPy**: Numerical computing library
- **Uvicorn**: ASGI server

## License

This project uses Apache 2.0 License. See PaddleOCR project for details: https://github.com/PaddlePaddle/PaddleOCR

## References

- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **PaddleX**: https://github.com/PaddlePaddle/PaddleX
- **FastAPI**: https://fastapi.tiangolo.com
- **Ollama**: https://ollama.ai

## Support

For issues, questions, or contributions:
- PaddleOCR Issues: https://github.com/PaddlePaddle/PaddleOCR/issues
- API Documentation: http://localhost:8001/docs (when running)

---

**Version**: 3.2.0  
**Last Updated**: October 2025

