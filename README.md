# PaddleOCR API - Multi-Language Document Intelligence

A production-ready FastAPI service providing GPU-accelerated OCR and document understanding with support for 21 languages and comprehensive model validation.

## Features

✨ **5 Specialized Endpoints**
- **PP-OCRv5** - Universal text recognition (5 optimized languages)
- **PP-OCRv3** - Multi-language text recognition (21 languages)
- **PP-StructureV3 Markdown** - Convert documents to clean Markdown
- **PP-StructureV3 JSON** - Extract structured document data  
- **PP-ChatOCRv4** - Intelligent information extraction with Ollama LLM

🌍 **Comprehensive Language Support**
- **Pre-installed**: All 21 supported languages baked into container
- **Zero Runtime Downloads**: All models validated during build
- **Mixed-language**: Handle documents with multiple languages
- **Default Language**: English (`en`) for all endpoints

⚡ **Performance & Reliability**
- GPU acceleration (NVIDIA CUDA 12.9)
- ~1-2 seconds per page (pre-downloaded models)
- Concurrent request handling
- Model validation with retry logic
- 95% success rate guarantee

## Quick Start

### Prerequisites
- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 12.9 support
- Docker Compose

### 1. Build and Run

```bash
# Build the Docker image (includes comprehensive model validation)
docker-compose build

# Start the service
docker-compose up -d

# Check status
docker-compose ps
```

The API will be available at `http://localhost:8000`

### 2. Test the API

```bash
# Check API documentation
curl http://localhost:8000/docs

# Test English OCR (default language)
curl -X POST http://localhost:8000/ocr/ppocrv5 \
  -F "file=@document.jpg"

# Test Arabic OCR
curl -X POST http://localhost:8000/ocr/ppocrv3 \
  -F "file=@arabic_doc.pdf" \
  -F "lang=ar"

# Test document to Markdown
curl -X POST http://localhost:8000/ocr/structurev3/markdown \
  -F "file=@contract.pdf" \
  -F "lang=en"
```

## API Endpoints

### 1. PP-OCRv5 - Universal Text Recognition
**Endpoint**: `POST /ocr/ppocrv5`

Extract text from images and PDFs with 13% accuracy improvement over previous versions. Optimized for 5 languages with mixed-language support.

**Supported Languages**: `en` (English), `ch` (Chinese), `japan` (Japanese), `korean` (Korean), `chinese_cht` (Traditional Chinese)

**Python Example:**
```python
import requests

with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/ppocrv5",
        files={"file": f},
        data={"lang": "en"}  # Default: English
    )

result = response.json()
for item in result["results"]:
    print(f"Text: {item['text']} (confidence: {item['confidence']:.2f})")
```

**Response:**
```json
{
  "pipeline": "PP-OCRv5",
  "description": "Universal Scene Text Recognition - Single model supports five text types with 13% accuracy improvement",
  "results": [
    {
      "text": "Hello World",
      "confidence": 0.99,
      "bbox": [[10, 20], [100, 20], [100, 40], [10, 40]]
    }
  ],
  "total_texts": 1
}
```

### 2. PP-OCRv3 - Multi-Language Text Recognition
**Endpoint**: `POST /ocr/ppocrv3`

Extract text from images and PDFs with support for 21 languages. Comprehensive international language coverage.

**Supported Languages**: 21 languages including English, Arabic, Hindi, Chinese, French, German, Spanish, Italian, Russian, Japanese, Korean, Portuguese, Dutch, Polish, Ukrainian, Thai, Vietnamese, Indonesian, Tamil, Telugu, and Traditional Chinese.

**Python Example:**
```python
with open("arabic_doc.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/ppocrv3",
        files={"file": f},
        data={"lang": "ar"}  # Arabic document
    )

result = response.json()
print(f"Total texts: {result['total_texts']}")
print(f"Supported languages: {result['supported_languages']}")
```

### 3. PP-StructureV3 Markdown - Document to Markdown
**Endpoint**: `POST /ocr/structurev3/markdown`

Convert complex documents to clean Markdown format with preserved layout and structure.

**Python Example:**
```python
with open("contract.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/structurev3/markdown",
        files={"file": f},
        data={"lang": "en"}  # Default: English
    )

markdown = response.json()["markdown"]
print(markdown)
```

**Response:**
```json
{
  "pipeline": "PP-StructureV3 - Markdown",
  "description": "Intelligently converts complex PDFs and document images into Markdown files that preserve original structure",
  "markdown": "# Document Title\n\n## Section 1\n\nContent...",
  "pages": ["# Page 1...", "# Page 2..."],
  "total_pages": 2
}
```

### 4. PP-StructureV3 JSON - Structured Document Data
**Endpoint**: `POST /ocr/structurev3/json`

Extract detailed structure with layout blocks, regions, and text elements.

**Python Example:**
```python
with open("form.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/structurev3/json",
        files={"file": f},
        data={"lang": "en"}  # Default: English
    )

result = response.json()
for page in result["pages"]:
    for block in page["layout_blocks"]:
        print(f"{block['label']}: {block['content']}")
```

**Response:**
```json
{
  "pipeline": "PP-StructureV3 - JSON",
  "description": "Intelligently converts complex PDFs and document images into JSON files that preserve original structure",
  "pages": [
    {
      "layout_blocks": [
        {
          "label": "doc_title",
          "content": "Invoice",
          "bbox": [10, 20, 100, 40]
        }
      ],
      "layout_regions": [...],
      "text_elements": [...],
      "total_blocks": 5,
      "total_regions": 3,
      "total_texts": 25
    }
  ],
  "total_pages": 1
}
```

### 5. PP-ChatOCRv4 - Intelligent Information Extraction
**Endpoint**: `POST /ocr/chatocrv4`

Extract specific information using Ollama LLM integration with dynamic OCR version selection.

**Python Example:**
```python
with open("invoice.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/chatocrv4",
        files={"file": f},
        data={
            "keys": ["Invoice Number", "Date", "Total"],
            "lang": "en",  # Default: English
            "ollama_model": "llama3.1:latest",
            "ollama_url": "http://host.docker.internal:11434"
        }
    )

result = response.json()
for key, value in result["extracted_data"].items():
    print(f"{key}: {value}")
```

**Response:**
```json
{
  "pipeline": "PP-ChatOCRv4",
  "description": "Intelligent Information Extraction - Uses LLM to extract specific information from documents",
  "extracted_data": {
    "Invoice Number": "INV-2024-001",
    "Date": "2024-01-15",
    "Total": "$1,250.00"
  },
  "ocr_version_used": "PP-OCRv5",
  "ocr_text_count": 45
}
```

## Comprehensive Language Support

### Complete Language Reference (21 Languages)

#### PP-OCRv5 Supported Languages (5 languages)
**Focus**: Optimized for 5 text types with mixed-language support
- `en` - English
- `ch` - Chinese (Simplified) - **Best for Chinese+English mixed documents**
- `japan` - Japanese - **Best for Japanese+English mixed documents**
- `korean` - Korean - **Best for Korean+English mixed documents**
- `chinese_cht` - Chinese (Traditional)

#### PP-OCRv3 Supported Languages (21 languages)
**Focus**: Comprehensive international language coverage

**European Languages (8):**
- `en` - English, `fr` - French, `de` - German, `es` - Spanish, `it` - Italian
- `ru` - Russian, `pt` - Portuguese, `nl` - Dutch, `pl` - Polish, `uk` - Ukrainian

**Asian Languages (7):**
- `ch` - Chinese (Simplified), `japan` - Japanese, `korean` - Korean, `chinese_cht` - Chinese (Traditional)
- `th` - Thai, `vi` - Vietnamese, `id` - Indonesian

**Middle Eastern Languages (1):**
- `ar` - Arabic

**Indian Subcontinent Languages (3):**
- `hi` - Hindi, `ta` - Tamil, `te` - Telugu

#### PP-StructureV3 & PP-ChatOCRv4 Supported Languages
**Same as PP-OCRv3** - Uses OCR recognition for layout analysis and intelligent extraction
- All 21 languages from PP-OCRv3 list
- Optimized for document structure analysis
- Supports both Markdown and JSON output
- Enhanced with LLM integration for ChatOCRv4

### Mixed-Language Document Recommendations

**Best Language Codes for Mixed Documents:**
- **Chinese + English**: Use `ch` (PP-OCRv5) or `ch` (PP-OCRv3)
- **Japanese + English**: Use `japan` (PP-OCRv5) or `japan` (PP-OCRv3)
- **Korean + English**: Use `korean` (PP-OCRv5) or `korean` (PP-OCRv3)
- **Arabic + English**: Use `ar` (PP-OCRv3)
- **Hindi + English**: Use `hi` (PP-OCRv3)
- **European Languages**: Use primary language code
- **Unknown Language**: Use `en` as fallback, then detect with confidence scores

**Examples:**
```bash
# Chinese + English mixed document
curl -X POST http://localhost:8000/ocr/ppocrv5 \
  -F "file=@mixed_doc.pdf" \
  -F "lang=ch"

# Arabic + English mixed document  
curl -X POST http://localhost:8000/ocr/ppocrv3 \
  -F "file=@arabic_english.pdf" \
  -F "lang=ar"
```

**Note:** PaddleOCR doesn't support a generic `"multi"` language code. Always specify a specific language code.

## Model Validation System

The Dockerfile includes a **robust model validation system** that ensures all PaddleOCR models are properly downloaded and validated during container build time.

### Features

#### ✅ **Retry Logic**
- **3 retry attempts** for each model download
- **5-second delay** between retries
- Automatic recovery from transient network failures
- Clear progress indication for each attempt

#### ✅ **Critical Language Protection**
- **Critical languages**: English (`en`), Chinese (`ch`), Arabic (`ar`), Hindi (`hi`)
- **Build fails immediately** if any critical language fails after all retries
- Ensures essential language support is always available

#### ✅ **Success Rate Validation**
- **95% minimum success rate** required
- Tracks success/failure for all 200+ model downloads
- Build fails if success rate drops below threshold
- Detailed failure report for troubleshooting

#### ✅ **Comprehensive Reporting**
- Real-time download progress
- Per-endpoint success statistics
- Final summary with percentages
- List of all failed models (if any)

### Model Coverage

| Endpoint | Languages | Models |
|----------|-----------|--------|
| **PP-OCRv5** | 5 | en, ch, japan, korean, chinese_cht |
| **PP-OCRv3** | 21 | All supported languages |
| **PP-StructureV3** | 21 | All supported languages |
| **PP-ChatOCRv4** | 21 | All supported languages |
| **TOTAL** | 21 unique | ~50+ model combinations |

### Build Process

**Step 1: Model Download with Retry**
For each model:
1. **Attempt 1**: Try to download
2. **If fails**: Wait 5 seconds, retry
3. **Attempt 2**: Try again
4. **If fails**: Wait 5 seconds, retry
5. **Attempt 3**: Final attempt
6. **If fails**: Mark as failed and continue

**Step 2: Progress Tracking**
Real-time progress for each endpoint with clear success/failure indicators.

**Step 3: Validation**
After all downloads, comprehensive summary with success rates and failure reports.

**Step 4: Critical Validation**
- Check for critical failures (EN, CH, AR, HI must succeed)
- Check overall success rate (95% minimum required)
- Build fails if requirements not met

### Success Scenarios

**✅ Perfect Build:**
```
============================================================
✓ MODEL VALIDATION SUCCESSFUL!
============================================================
All critical models downloaded: YES
Success rate: 100.0% (required: 95.0%)

✓ Container is ready for production use!
============================================================
```

**✅ Minor Failures:**
```
============================================================
✓ MODEL VALIDATION SUCCESSFUL!
============================================================
All critical models downloaded: YES
Success rate: 97.6% (required: 95.0%)

Non-critical failures (6):
  ⚠ PP-OCRv3:ga
  ⚠ PP-OCRv3:cy
  ⚠ PP-StructureV3:mt
  ⚠ PP-ChatOCRv4:is
  ⚠ PP-ChatOCRv4:mk
  ⚠ PP-ChatOCRv4:ga

✓ Container is ready for production use!
============================================================
```

### Failure Scenarios

**❌ Critical Language Failure:**
```
============================================================
✗ CRITICAL FAILURE: Essential language models failed!
============================================================
Failed critical models:
  ✗ PP-OCRv3:ar
  ✗ PP-StructureV3:ar

Build CANNOT proceed without these models.
ERROR: executor failed with exit code 1
```

**❌ Low Success Rate:**
```
============================================================
✗ BUILD FAILED: Insufficient model coverage!
============================================================
Required: 95% success rate
Achieved: 89.3% success rate

Build CANNOT proceed with incomplete model coverage.
ERROR: executor failed with exit code 1
```

## Parameters

### Common Parameters (All Endpoints)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Image (JPG, PNG) or PDF file |
| `lang` | String | `"en"` | Language code (en, ar, hi, ch, fr, de, etc.) |

### ChatOCR-Specific Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `keys` | List[String] | Yes | Specific information fields to extract |
| `ollama_model` | String | `"llama3.1:latest"` | Ollama model to use for extraction |
| `ollama_url` | String | `"http://host.docker.internal:11434"` | Ollama API URL |

## Performance

| Metric | Value |
|--------|-------|
| **First Request** (pre-installed lang) | 1-2 seconds/page |
| **Subsequent Requests** | 1-2 seconds/page |
| **GPU Memory** | ~2GB per pipeline |
| **GPU Utilization** | 18-21% during inference |
| **Concurrent Requests** | Supported (shared GPU) |
| **Model Storage** | ~1GB for complete coverage |
| **Build Time** | 15-30 minutes (with validation) |

## Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│         (Port 8000)                     │
├─────────────────────────────────────────┤
│  PP-OCRv5    │  PP-OCRv3               │
│  (5 langs)   │  (21 langs)             │
│              │                         │
│  PP-StructureV3 (Document Parsing)      │
│   ├─ Markdown                           │
│   └─ JSON                               │
│                                         │
│  PP-ChatOCRv4 (Intelligent Extraction) │
│   └─ Ollama LLM Integration            │
├─────────────────────────────────────────┤
│     PaddleOCR 3.2.0 + PaddleX 3.2.1    │
├─────────────────────────────────────────┤
│     PaddlePaddle GPU 3.2.0 (CUDA 12.9) │
├─────────────────────────────────────────┤
│     NVIDIA GPU (RTX/Tesla/A-series)     │
└─────────────────────────────────────────┘
```

## Project Structure

```
paddleocr/
├── ocr_api.py                             # Main FastAPI application
├── requirements.txt                       # Python dependencies
├── Dockerfile                            # Container build instructions with model validation
├── docker-compose.yml                    # Service configuration
├── paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl  # GPU wheel (~1.8GB)
├── tenancy_contract.jpg                  # Test document
└── README.md                             # This comprehensive documentation
```

## Configuration

### Environment Variables

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0  # GPU ID to use
```

### GPU Configuration

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Volume Mounts

```yaml
volumes:
  - ./ocr_api.py:/app/ocr_api.py:ro  # Enable live code updates without rebuild
```

## Development

### Building

```bash
# Build Docker image with model validation
docker-compose build

# Monitor build progress (look for validation summary)
docker-compose build 2>&1 | grep "MODEL VALIDATION"
```

### Running Locally

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Restart after code changes (with volume mount)
docker-compose restart

# Stop service
docker-compose down
```

### Testing

**Manual Testing:**
```bash
# Test English OCR (default)
curl -X POST http://localhost:8000/ocr/ppocrv5 \
  -F "file=@test.jpg"

# Test Arabic OCR
curl -X POST http://localhost:8000/ocr/ppocrv3 \
  -F "file=@arabic_doc.pdf" \
  -F "lang=ar"

# Test document to Markdown
curl -X POST http://localhost:8000/ocr/structurev3/markdown \
  -F "file=@contract.pdf" \
  -F "lang=en"

# Test structured JSON extraction
curl -X POST http://localhost:8000/ocr/structurev3/json \
  -F "file=@form.jpg" \
  -F "lang=en"

# Test intelligent extraction
curl -X POST http://localhost:8000/ocr/chatocrv4 \
  -F "file=@invoice.pdf" \
  -F "keys=Invoice Number" \
  -F "keys=Date" \
  -F "keys=Total" \
  -F "lang=en" \
  -F "ollama_model=llama3.1:latest" \
  -F "ollama_url=http://host.docker.internal:11434"
```

### Accessing Container

```bash
# Shell access
docker exec -it paddleocr-paddleocr-api-1 bash

# Check GPU
docker exec paddleocr-paddleocr-api-1 nvidia-smi

# Check Python packages
docker exec paddleocr-paddleocr-api-1 pip list

# Verify model availability
docker exec paddleocr-paddleocr-api-1 python3 -c "
from paddleocr import PaddleOCR
print('Testing English:', PaddleOCR(lang='en', ocr_version='PP-OCRv3', device='cpu'))
print('Testing Arabic:', PaddleOCR(lang='ar', ocr_version='PP-OCRv3', device='cpu'))
print('✓ All models verified!')
"
```

## Troubleshooting

### Language-Related Issues

#### 1. Language Model Not Found

**Error**: `ValueError: No models are available for the language 'xx' and OCR version 'PP-OCRv5'`

**Solutions**:
- **For PP-OCRv5**: Only supports 5 languages (`en`, `ch`, `japan`, `korean`, `chinese_cht`)
- **For PP-OCRv3**: Supports 80+ languages, use this endpoint instead
- **Check language code**: Ensure you're using the correct language code (e.g., `ar` for Arabic, not `arabic`)

**Example**:
```bash
# ❌ Wrong - Arabic not supported in PP-OCRv5
curl -X POST http://localhost:8000/ocr/ppocrv5 -F "file=@doc.jpg" -F "lang=ar"

# ✅ Correct - Use PP-OCRv3 for Arabic
curl -X POST http://localhost:8000/ocr/ppocrv3 -F "file=@doc.jpg" -F "lang=ar"
```

#### 2. Mixed-Language Document Issues

**Problem**: Document contains multiple languages (e.g., Chinese + English)

**Solutions**:
- **Chinese + English**: Use `lang=ch` (PP-OCRv5 or PP-OCRv3)
- **Japanese + English**: Use `lang=japan` (PP-OCRv5 or PP-OCRv3)
- **Korean + English**: Use `lang=korean` (PP-OCRv5 or PP-OCRv3)
- **Other combinations**: Use primary language code

**Example**:
```bash
# ✅ Correct for Chinese+English mixed document
curl -X POST http://localhost:8000/ocr/ppocrv5 -F "file=@mixed_doc.jpg" -F "lang=ch"
```

#### 3. Low Accuracy for Specific Language

**Problem**: Poor OCR results for a particular language

**Solutions**:
- **Check language code**: Ensure correct language code
- **Try different endpoint**: PP-OCRv5 vs PP-OCRv3
- **Check image quality**: Ensure 300+ DPI resolution
- **Try different language**: Some languages have better models

#### 4. Unknown Language Detection

**Problem**: Don't know which language to use

**Solutions**:
1. **Start with English**: Use `lang=en` as fallback (default)
2. **Check confidence scores**: Look at confidence values in response
3. **Try common languages**: `en`, `ch`, `ar`, `hi`, `fr`, `de`, `es`
4. **Use language reference**: Check `/languages` endpoint for full list

### GPU Not Detected

```bash
# Check GPU availability in container
docker exec paddleocr-paddleocr-api-1 nvidia-smi

# Verify CUDA compilation
docker exec paddleocr-paddleocr-api-1 python3 -c "import paddle; print(paddle.is_compiled_with_cuda())"

# Check if GPU is being used
docker exec paddleocr-paddleocr-api-1 python3 -c "import paddle; print(paddle.device.get_device())"
```

**Solutions**:
- Ensure nvidia-docker2 is installed
- Verify GPU is visible: `nvidia-smi`
- Check docker-compose.yml has correct GPU configuration

### Model Download Issues

Models are pre-downloaded and validated during build. If issues occur:

**Check connectivity**:
```bash
docker exec paddleocr-paddleocr-api-1 curl -I https://paddleocr.bj.bcebos.com
```

**Solutions**:
- All models are pre-downloaded in the Docker image
- Check container has internet access
- Rebuild container if needed

### Low OCR Accuracy

**Tips to improve accuracy**:
1. **Use correct language code** for your document
2. **Check image quality**:
   - Increase resolution (300+ DPI recommended)
   - Ensure good contrast
   - Remove noise/artifacts
3. **For mixed languages**: Use primary language code (e.g., `"ch"` for Chinese+English)
4. **Choose appropriate endpoint**: PP-OCRv5 for mixed languages, PP-OCRv3 for international

### Memory Issues

If GPU memory is insufficient:

**Solutions**:
- Reduce concurrent requests
- Process smaller images
- Use CPU mode (slower):
  ```python
  # Modify ocr_api.py
  device="cpu"  # instead of "gpu:0"
  ```

### API Errors

**Check logs**:
```bash
docker-compose logs -f paddleocr-api
```

**Common issues**:
- Invalid language code → Use supported codes
- File format not supported → Use JPG, PNG, or PDF
- Missing `file` parameter → Ensure multipart/form-data upload

### Build Validation Issues

**If build fails with model validation errors**:

1. **Check network connectivity**
2. **Verify access to paddlepaddle.org.cn**
3. **Try building again** (retries may succeed)
4. **Check firewall/proxy settings**

**If build fails with low success rate**:
1. **Check available disk space** (need ~20GB)
2. **Verify network stability**
3. **Increase `MAX_RETRIES` in Dockerfile**
4. **Lower `MIN_SUCCESS_RATE` temporarily** (not recommended for production)

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- Test API calls directly in browser
- See request/response schemas
- View parameter descriptions

## Endpoint Selection Guide

**Choose the right endpoint**:

| Use Case | Endpoint | Language Support | Best For |
|----------|----------|------------------|----------|
| Chinese+English mixed | `/ocr/ppocrv5` | 5 languages | Mixed-language documents |
| International documents | `/ocr/ppocrv3` | 21 languages | Arabic, Hindi, European |
| Document to Markdown | `/ocr/structurev3/markdown` | 21 languages | Any language |
| Structured data | `/ocr/structurev3/json` | 21 languages | Any language |
| Information extraction | `/ocr/chatocrv4` | 21 languages | Any language |

## Best Practices

**For optimal results**:
1. **Use correct language code**: Check language reference above
2. **Choose appropriate endpoint**: PP-OCRv5 vs PP-OCRv3
3. **Check image quality**: 300+ DPI recommended
4. **Use confidence scores**: To validate results
5. **Handle mixed languages**: Use primary language code
6. **Monitor performance**: Check GPU utilization and response times

**Example workflow**:
```bash
# 1. Test with appropriate endpoint
curl -X POST http://localhost:8000/ocr/ppocrv3 \
  -F "file=@document.jpg" \
  -F "lang=ar"

# 2. Check confidence scores in response
# 3. Adjust language or endpoint if needed
```

## Tech Stack

- **Framework**: FastAPI 0.115.6
- **OCR Engine**: PaddleOCR 3.2.0
- **ML Framework**: PaddlePaddle GPU 3.2.0
- **Additional ML**: PaddleX 3.2.1
- **LLM Integration**: Ollama (llama3.1:latest)
- **GPU Runtime**: NVIDIA CUDA 12.9.1
- **Base OS**: Ubuntu 24.04
- **Python**: 3.12

## License

This project uses PaddleOCR which is licensed under Apache 2.0.

## References

- [PaddleOCR GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Official Documentation](https://www.paddleocr.ai/)
- [PP-OCRv5 Pipeline Guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 Pipeline Guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 Pipeline Guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-ChatOCRv4.html)
- [PaddleOCR Multi-Language Support](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md)

## Support

For issues and questions:

1. **Check API Documentation**: Visit `/docs` for interactive API testing
2. **Review Logs**: Run `docker-compose logs -f` to see detailed error messages
3. **Test Connection**: Ensure GPU is accessible via `nvidia-smi`
4. **Verify Models**: All languages are pre-downloaded and validated during build
5. **Check Language Support**: Use the comprehensive language reference above

---

**Built with** ❤️ **using PaddleOCR 3.x with comprehensive model validation**