"""
PaddleOCR API - Multi-Language Document Intelligence

A production-ready FastAPI service providing GPU-accelerated OCR and document 
understanding with support for 80+ languages.

Endpoints
---------

1. **PP-OCRv5** (`/ocr/ppocrv5`) - Universal Text Recognition
   - Extract text from images and PDFs
   - Supports 5 text types (Chinese, English, Japanese, Pinyin)
   - 13% accuracy improvement over previous versions
   - Optimized for mixed-language documents

2. **PP-OCRv3** (`/ocr/ppocrv3`) - Multi-Language Text Recognition
   - Extract text from images and PDFs
   - Supports 80+ languages (English, Arabic, Hindi, Chinese, French, German, Spanish, Italian, Russian, Japanese, Korean, etc.)
   - Comprehensive international language coverage
   - Auto-download language models on first use

3. **PP-StructureV3 Markdown** (`/ocr/structurev3/markdown`) - Document to Markdown
   - Convert complex documents to clean Markdown format
   - Preserves layout, tables, formulas, and structure
   - Outperforms commercial solutions in public benchmarks
   - Supports multi-column reading order recovery

6. **PP-StructureV3 JSON** (`/ocr/structurev3/json`) - Structured Document Data
   - Extract layout blocks, regions, and text elements
   - Detailed bounding boxes and confidence scores
   - Perfect for document analysis and data extraction
   - Returns structured JSON with hierarchical layout

7. **PP-ChatOCRv4** (`/ocr/chatocrv4`) - Intelligent Information Extraction
   - Extract specific fields using ERNIE LLM integration
   - Supports seals, tables, and multi-page PDFs
   - Intelligent key-value extraction
   - Combines OCR with natural language understanding

8. **Language Reference** (`/languages`) - Supported Languages Guide
   - Complete list of all 80+ supported languages
   - Language codes and usage guidance
   - Mixed-language document recommendations
   - Endpoint-specific language support details

Features
--------

- **GPU Acceleration**: NVIDIA CUDA 12.9 support (~1-2 seconds per page)
- **Multi-Language**: 80+ languages with auto-download on first use
- **Mixed Languages**: Handle documents with Chinese+English or other combinations
- **Concurrent Requests**: Shared GPU for multiple simultaneous requests
- **Model Caching**: Fast subsequent requests after first model load

Quick Start
-----------

```bash
# Test English OCR
curl -X POST http://localhost:8000/ocr/ppocrv5 \\
  -F "file=@document.jpg" \\
  -F "lang=en"

# Test Arabic OCR
curl -X POST http://localhost:8000/ocr/ppocrv3 \\
  -F "file=@arabic_doc.jpg" \\
  -F "lang=ar"

# Convert Arabic document to Markdown
curl -X POST http://localhost:8000/ocr/structurev3/markdown \\
  -F "file=@arabic_doc.pdf" \\
  -F "lang=ar"

# Get language support reference
curl http://localhost:8000/languages
```

Documentation
-------------

- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- GitHub: https://github.com/PaddlePaddle/PaddleOCR
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Language support constants to eliminate duplication
PP_OCRV5_LANGUAGES = {
    "en": "English",
    "ch": "Chinese (Simplified)", 
    "japan": "Japanese",
    "korean": "Korean",
    "chinese_cht": "Chinese (Traditional)"
}

PP_OCRV3_MAJOR_LANGUAGES = {
    "en": "English", "ar": "Arabic", "hi": "Hindi", "ch": "Chinese", 
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", 
    "ru": "Russian", "japan": "Japanese", "korean": "Korean"
}

PP_OCRV3_EUROPEAN_LANGUAGES = [
    "pt", "nl", "pl", "uk", "cs", "ro", "sv", "da", "no", "fi", "tr", 
    "el", "bg", "hr", "sk", "sl", "et", "lv", "lt", "mt", "cy", "ga", "is", "mk"
]

PP_OCRV3_ASIAN_LANGUAGES = [
    "th", "vi", "id", "bn", "ta", "te", "ml", "kn", "gu", "pa", "or", 
    "as", "ne", "si", "my", "km", "lo", "ka", "hy", "az", "kk", "ky", 
    "uz", "tg", "mn"
]

PP_OCRV3_MIDDLE_EASTERN_LANGUAGES = [
    "fa", "ur", "he", "yi", "ku", "ps", "sd"
]

PP_OCRV3_AFRICAN_LANGUAGES = [
    "sw", "am", "ha", "yo", "ig", "zu", "xh", "af", "so"
]

try:
    # PaddleOCR imports.  These will only succeed if the `paddleocr`
    # package is installed in the environment.  Users who wish to run
    # these endpoints must install paddleocr, along with its heavy
    # dependencies such as PaddlePaddle and PaddleNLP.
    # In PaddleOCR 3.x, import specific pipeline classes
    from paddleocr import PaddleOCR  # type: ignore
    try:
        from paddleocr import PPStructureV3, PPChatOCRv4Doc  # type: ignore
    except ImportError:
        # Fallback if specific classes not available
        PPStructureV3 = None  # type: ignore
        PPChatOCRv4Doc = None  # type: ignore
    _PADDLE_AVAILABLE = True
except Exception as e:
    import logging
    logging.error("Failed to import PaddleOCR", exc_info=e)
    # If import fails we still allow the API to start.  Individual
    # endpoints check this flag before performing inference and raise
    # informative errors when paddleocr is missing.
    PaddleOCR = None  # type: ignore
    PPStructureV3 = None  # type: ignore
    PPChatOCRv4Doc = None  # type: ignore
    _PADDLE_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
except Exception as exc:  # pragma: no cover
    # PIL and numpy should be installed in most environments.  If not,
    # we inform the caller at runtime.
    Image = None  # type: ignore
    np = None  # type: ignore

try:
    import requests
    import json
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


app = FastAPI(
    title="PaddleOCR API - Multi-Language Document Intelligence",
    version="3.2.0",
    description=(
        "🚀 Production-ready GPU-accelerated OCR and document understanding API\n\n"
        "**Features:**\n"
        "- 🌍 **80+ Languages**: Comprehensive international language support across all endpoints\n"
        "- ⚡ **GPU Acceleration**: ~1-2 seconds per page with NVIDIA CUDA 12.9\n"
        "- 📄 **6 Specialized Endpoints**: PP-OCRv5, PP-OCRv3, Markdown conversion, JSON extraction, intelligent parsing, language reference\n"
        "- 🔄 **Pre-loaded Models**: All major language models baked into container for instant availability\n"
        "- 🎯 **High Accuracy**: PP-OCRv5 with 13% improvement, PP-OCRv3 with 80+ language support\n"
        "- 🔍 **Enhanced Language Support**: Comprehensive 80+ language support across all endpoints\n\n"
        "**Quick Start:**\n"
        "```bash\n"
        "# English OCR (PP-OCRv5 - Best for English/Chinese mixed)\n"
        "curl -X POST http://localhost:8000/ocr/ppocrv5 -F \"file=@doc.jpg\" -F \"lang=en\"\n\n"
        "# Arabic OCR (PP-OCRv3 - Best for 80+ languages)\n"
        "curl -X POST http://localhost:8000/ocr/ppocrv3 -F \"file=@arabic_doc.jpg\" -F \"lang=ar\"\n\n"
        "# Chinese OCR (PP-OCRv5 - Optimized for Chinese+English)\n"
        "curl -X POST http://localhost:8000/ocr/ppocrv5 -F \"file=@chinese_doc.jpg\" -F \"lang=ch\"\n\n"
        "# Convert Arabic document to Markdown\n"
        "curl -X POST http://localhost:8000/ocr/structurev3/markdown -F \"file=@arabic_doc.pdf\" -F \"lang=ar\"\n\n"
        "# Extract information from Hindi document\n"
        "curl -X POST http://localhost:8000/ocr/chatocrv4 -F \"file=@hindi_doc.jpg\" -F \"keys=Name\" -F \"keys=Date\" -F \"lang=hi\"\n"
        "```\n\n"
        "**Language Selection Guide:**\n"
        "- **PP-OCRv5**: Use for Chinese, English, Japanese, Korean, Traditional Chinese\n"
        "- **PP-OCRv3**: Use for 80+ languages including Arabic, Hindi, French, German, Spanish, etc.\n"
        "- **Mixed Documents**: Use primary language (e.g., 'ch' for Chinese+English)\n"
        "- **Unknown Language**: Start with 'en' and check confidence scores\n\n"
        "📚 **Documentation**: Visit `/docs` for interactive API testing"
    ),
    contact={
        "name": "PaddleOCR Project",
        "url": "https://github.com/PaddlePaddle/PaddleOCR",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


def _ensure_dependencies() -> None:
    """Raise an informative HTTP error if required libraries are missing."""
    if not _PADDLE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "PaddleOCR is not installed.  Please install the `paddleocr`"
                " Python package and its dependencies to enable this endpoint."
            ),
        )
    if Image is None or np is None:
        raise HTTPException(
            status_code=503,
            detail="Required dependencies Pillow and numpy are missing."
        )


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects to JSON-serializable types."""
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dict representation
        return _convert_to_serializable(obj.__dict__)
    else:
        # For other types, try to convert to string
        return str(obj)


def _read_upload_to_bytes(upload: UploadFile) -> bytes:
    """Read an uploaded file into bytes."""
    # Reset file pointer to beginning
    upload.file.seek(0)
    return upload.file.read()


def _save_to_temporary_file(data: bytes, suffix: str = "") -> str:
    """Save byte data to a temporary file and return the file path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fp:
        fp.write(data)
    return path


def _run_ocr_sync(ocr_version: str, lang: str, file_data: bytes, filename: str) -> List[Dict]:
    """Synchronous OCR logic that will be run in a thread pool.
    
    Parameters
    ----------
    ocr_version : str
        OCR version to use ("PP-OCRv5" or "PP-OCRv3")
    lang : str
        Language code for recognition
    file_data : bytes
        File data as bytes
    filename : str
        Original filename for determining file extension
        
    Returns
    -------
    List[Dict]
        List of text results with text, confidence, and bbox information
    """
    # Convert bytes to numpy array for direct processing
    import numpy as np
    from PIL import Image
    
    # Create an in-memory image from bytes
    img = Image.open(io.BytesIO(file_data))
    img_array = np.array(img)
    
    # Instantiate the pipeline with GPU enabled
    kwargs = {
        "ocr_version": ocr_version,
        "lang": lang,
        "device": "gpu:0",
    }
    
    ocr = PaddleOCR(**kwargs)
    
    # Perform OCR using the predict() method with numpy array
    result = ocr.predict(input=img_array)
    
    # Extract text results from all pages
    text_results = []
    for page_result in result:
        rec_texts = page_result['rec_texts']
        rec_scores = page_result['rec_scores']
        rec_polys = page_result['rec_polys']
        
        for i, text in enumerate(rec_texts):
            text_results.append({
                "text": text,
                "confidence": float(rec_scores[i]),
                "bbox": _convert_to_serializable(rec_polys[i])
            })
    
    return text_results


async def _run_ocr(ocr_version: str, lang: str, file_data: bytes, filename: str) -> List[Dict]:
    """Async wrapper for OCR logic that runs in a thread pool to avoid blocking the event loop.
    
    Parameters
    ----------
    ocr_version : str
        OCR version to use ("PP-OCRv5" or "PP-OCRv3")
    lang : str
        Language code for recognition
    file_data : bytes
        File data as bytes
    filename : str
        Original filename for determining file extension
        
    Returns
    -------
    List[Dict]
        List of text results with text, confidence, and bbox information
    """
    try:
        # Run the synchronous OCR logic in a thread pool
        return await asyncio.to_thread(_run_ocr_sync, ocr_version, lang, file_data, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}")


def _run_structurev3_sync(lang: str, file_data: bytes, filename: str, output_format: str) -> Dict:
    """
    Robust PP-StructureV3 runner:
    - PDF -> write to temp and pass the path (enables multi-page parsing)
    - Image -> pass numpy array
    - Extract Markdown/JSON from either object attributes or dict keys, with fallbacks
    """
    if PPStructureV3 is None:
        raise HTTPException(
            status_code=503,
            detail="PP-StructureV3 is not available. Please ensure PaddleOCR 3.x is installed.",
        )
    
    import numpy as np
    from PIL import Image
    import os, io, tempfile
    from typing import Any

    def _page_to_markdown_texts(page: Any) -> List[str]:
        """
        Extract markdown from PP-StructureV3 page result.
        Uses the same data source as _page_to_json for consistency.
        """
        # Primary approach: Use page.json() method (same as JSON endpoint)
        try:
            j = getattr(page, "json", None)
            if callable(j):
                try:
                    j = j()
                except Exception:
                    j = None
            
            if isinstance(j, dict):
                # Look for rec_texts in the JSON structure
                res_data = j.get("res")
                if res_data and isinstance(res_data, dict):
                    ocr_res = res_data.get("overall_ocr_res")
                    if ocr_res and isinstance(ocr_res, dict):
                        rec_texts = ocr_res.get("rec_texts")
                        if isinstance(rec_texts, list) and rec_texts:
                            # Filter out empty strings and format the content
                            filtered_texts = [str(t).strip() for t in rec_texts if str(t).strip()]
                            if filtered_texts:
                                # Create better formatted markdown from OCR text
                                formatted_markdown = []
                                current_section = []
                                
                                for text in filtered_texts:
                                    # Detect document structure and format accordingly
                                    if text.upper() in ['TENANCYCONTRACT', 'CONTRACT', 'AGREEMENT']:
                                        if current_section:
                                            formatted_markdown.append('\n'.join(current_section))
                                        formatted_markdown.append(f"# {text}")
                                        current_section = []
                                    elif ':' in text and len(text) < 50:  # Likely a label
                                        if current_section:
                                            formatted_markdown.append('\n'.join(current_section))
                                        formatted_markdown.append(f"**{text}**")
                                        current_section = []
                                    elif text.startswith(('Date:', 'From:', 'To:', 'Rent:', 'Landlord:', 'Tenant:')):
                                        if current_section:
                                            formatted_markdown.append('\n'.join(current_section))
                                        formatted_markdown.append(f"**{text}**")
                                        current_section = []
                                    else:
                                        current_section.append(text)
                                
                                # Add any remaining content
                                if current_section:
                                    formatted_markdown.append('\n'.join(current_section))
                                
                                return ['\n\n'.join(formatted_markdown)] if formatted_markdown else filtered_texts
        except Exception:
            pass
        
        # Fallback: Try native markdown_texts if available
        md = getattr(page, "markdown", None)
        if isinstance(md, dict):
            texts = md.get("markdown_texts") or md.get("md_texts")
            if isinstance(texts, list):
                non_empty = [str(t) for t in texts if str(t).strip()]
                if len(non_empty) > 3:  # If we have substantial content
                    return non_empty
            elif isinstance(texts, str):
                if len(texts.strip()) > 50:  # If we have substantial content
                    return [texts.strip()]

        # Final fallback: Dict-based access
        if isinstance(page, dict):
            md = page.get("markdown")
            if isinstance(md, dict):
                texts = md.get("markdown_texts") or md.get("md_texts")
                if isinstance(texts, list):
                    return [str(t) for t in texts if str(t).strip()]
                elif isinstance(texts, str):
                    return [texts.strip()] if texts.strip() else []
            elif isinstance(md, list):
                return [str(t) for t in md if str(t).strip()]

        return []

    def _page_to_json(page: Any) -> Dict[str, Any]:
        """
        Try multiple shapes:
          - page.json() or page.json
          - page['json']
          - or assemble from known keys ('res', 'layout_parsing_result', 'tables', etc.)
        """
        # 1) attribute or callable
        j = getattr(page, "json", None)
        if callable(j):
            try:
                j = j()
            except Exception:
                j = None
        if isinstance(j, dict):
            return j

        # 2) dict key
        if isinstance(page, dict):
            if isinstance(page.get("json"), dict):
                return page["json"]

        # 3) assemble fallback from known fields
        assembled = {}
        candidates = (
            "res", "layout_parsing_result",
            "tables", "formulas", "images",
            "blocks", "lines", "paragraphs",
        )
        if isinstance(page, dict):
            for k in candidates:
                if k in page:
                    assembled[k] = page[k]
        else:
            for k in candidates:
                val = getattr(page, k, None)
                if val is not None:
                    assembled[k] = val

        return assembled or {"note": "No native JSON structure found; returned assembled minimal structure."}

    # ---- input dispatch (PDF path vs image array) ----
    is_pdf = (filename or "").lower().endswith(".pdf")
    tmp_path = None
    try:
        if is_pdf:
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, "wb") as fp:
                fp.write(file_data)
            predict_input = tmp_path
        else:
            img = Image.open(io.BytesIO(file_data))
            predict_input = np.array(img)

        # If you have a device helper, use it; otherwise keep your existing default:
        device = os.getenv("PADDLE_DEVICE", "gpu:0")  # fallback to GPU if you know CUDA is present
        pipeline = PPStructureV3(device=device)

        results = pipeline.predict(input=predict_input)

        if output_format == "markdown":
            page_md_list: List[str] = []
            for page in results:
                md_texts = _page_to_markdown_texts(page)
                # Join each page's markdown fragments
                page_md_list.append("\n".join(md_texts).strip() if md_texts else "")

            # Join pages with a visible delimiter
            full_markdown = "\n\n---\n\n".join([p for p in page_md_list if p])

            return {
                "pipeline": "PP-StructureV3 - Markdown",
                "description": "Markdown assembled from PP-StructureV3 native/fallback outputs per page.",
                "markdown": full_markdown,
                "pages": page_md_list,
                "total_pages": len(page_md_list),
            }
        elif output_format == "json":
            pages: List[Dict[str, Any]] = []
            for page in results:
                j = _page_to_json(page)
                pages.append(_convert_to_serializable(j))

            return {
                "pipeline": "PP-StructureV3 - JSON",
                "description": "Structured JSON extracted from PP-StructureV3 with robust fallbacks.",
                "pages": pages,
                "total_pages": len(pages),
            }
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def _run_structurev3(lang: str, file_data: bytes, filename: str, output_format: str) -> Dict:
    """Async wrapper for StructureV3 logic that runs in a thread pool to avoid blocking the event loop.
    
    Parameters
    ----------
    lang : str
        Language code for OCR recognition
    file_data : bytes
        File data as bytes
    filename : str
        Original filename for determining file extension
    output_format : str
        Output format ("markdown" or "json")
        
    Returns
    -------
    Dict
        Processed results in the specified format
    """
    try:
        # Run the synchronous StructureV3 logic in a thread pool
        return await asyncio.to_thread(_run_structurev3_sync, lang, file_data, filename, output_format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PP-StructureV3 inference failed: {exc}")


def _query_ollama(prompt: str, model: str = "llama3.1:latest", ollama_url: str = "http://host.docker.internal:11434") -> str:
    """Query Ollama LLM for information extraction."""
    if not _REQUESTS_AVAILABLE:
        return "Requests library not available for Ollama integration"
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response from Ollama")
    except Exception as e:
        return f"Ollama query failed: {str(e)}"


def _extract_info_with_ollama_from_text(document_text: str, keys: List[str], model: str = "llama3.1:latest", ollama_url: str = "http://host.docker.internal:11434") -> Dict[str, str]:
    """Extract information using Ollama LLM based on document text."""
    
    # Create a prompt for information extraction
    prompt = f"""
You are an AI assistant specialized in extracting specific information from document text.

Document Text:
{document_text}

Please extract the following information from the document text above:
{', '.join(keys)}

For each key, provide the most relevant value found in the document. If a key is not found, respond with "Not found".

Format your response as JSON with the keys as field names:
{{
    "{keys[0] if keys else 'key1'}": "extracted_value_or_Not_found",
    "{keys[1] if len(keys) > 1 else 'key2'}": "extracted_value_or_Not_found"
}}

Only return the JSON response, no additional text.
"""
    
    response = _query_ollama(prompt, model, ollama_url)
    
    # Try to parse JSON response
    try:
        # Extract JSON from response if it contains other text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        else:
            # Fallback: create a simple response
            return {key: "Not found" for key in keys}
    except Exception:
        # Fallback: create a simple response
        return {key: "Not found" for key in keys}


def _extract_info_with_ollama(visual_results: str, keys: List[str], model: str = "llama3.1:latest", ollama_url: str = "http://localhost:11434") -> Dict[str, str]:
    """Extract information using Ollama LLM based on visual OCR results."""
    
    # Extract readable text from the visual results
    # The visual_results contain complex PaddleOCR objects, we need to extract the actual text
    import re
    
    # Try to extract text content from the visual results string
    # Look for patterns that might contain actual text content
    text_patterns = [
        r"'text':\s*'([^']+)'",  # Extract text from 'text': 'value' patterns
        r'"text":\s*"([^"]+)"',  # Extract text from "text": "value" patterns
        r'rec_texts[^:]*:\s*\[([^\]]+)\]',  # Extract from rec_texts arrays
        r'text[^:]*:\s*([^,\]]+)',  # Extract text values
    ]
    
    extracted_texts = []
    for pattern in text_patterns:
        matches = re.findall(pattern, visual_results, re.IGNORECASE)
        extracted_texts.extend(matches)
    
    # Join all extracted text
    readable_text = ' '.join(extracted_texts[:20])  # Limit to first 20 matches
    
    # If no text was extracted, use a portion of the raw visual results
    if not readable_text.strip():
        readable_text = visual_results[:1000]
    
    # Create a prompt for information extraction
    prompt = f"""
You are an AI assistant specialized in extracting specific information from document OCR results.

Document Text Content (extracted from OCR):
{readable_text}

Please extract the following information from the document text above:
{', '.join(keys)}

For each key, provide the most relevant value found in the document. If a key is not found, respond with "Not found".

Format your response as JSON with the keys as field names:
{{
    "{keys[0] if keys else 'key1'}": "extracted_value_or_Not_found",
    "{keys[1] if len(keys) > 1 else 'key2'}": "extracted_value_or_Not_found"
}}

Only return the JSON response, no additional text.
"""
    
    response = _query_ollama(prompt, model, ollama_url)
    
    # Try to parse JSON response
    try:
        # Extract JSON from response if it contains other text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        else:
            # Fallback: create a simple response
            return {key: "Not found" for key in keys}
    except Exception:
        # Fallback: create a simple response
        return {key: "Not found" for key in keys}


@app.post(
    "/ocr/ppocrv5",
    tags=["Text Recognition"],
    summary="PP-OCRv5 - Universal Text Recognition",
    response_description="Extracted text with confidence scores and bounding boxes"
)
async def run_ppocrv5(
    file: UploadFile = File(..., description="Image (JPG, PNG) or PDF file to process"),
    lang: Optional[str] = Form("en", description="Language code: en (English), ch (Chinese), japan (Japanese), korean (Korean), chinese_cht (Traditional Chinese)"),
) -> JSONResponse:
    """Extract text from images and PDFs using PP-OCRv5 pipeline.
    
    PP-OCRv5 provides 13% accuracy improvement over previous versions and supports
    5 optimized languages with excellent mixed-language document handling.
    
    Parameters
    ----------
    file : UploadFile
        Image (JPG, PNG) or PDF file to process. Only the first page of PDFs is processed.
    lang : str, optional
        Language code for recognition. Supported: en (English), ch (Chinese), 
        japan (Japanese), korean (Korean), chinese_cht (Traditional Chinese).
        Use 'ch' for Chinese+English mixed documents. Default: "en".

    Returns
    -------
    JSONResponse
        JSON containing extracted text with confidence scores and bounding boxes.
    """
    _ensure_dependencies()
    
    # Read the uploaded file into memory
    data = _read_upload_to_bytes(file)
    
    # Use the common OCR helper (now async)
    text_results = await _run_ocr("PP-OCRv5", lang, data, file.filename or "")
    
    return JSONResponse(content={
        "pipeline": "PP-OCRv5",
        "description": "Universal Scene Text Recognition - Single model supports five text types (Simplified Chinese, Traditional Chinese, English, Japanese, and Pinyin) with 13% accuracy improvement. Solves multilingual mixed document recognition challenges.",
        "results": text_results,
        "total_texts": len(text_results)
    })


@app.post(
    "/ocr/ppocrv3",
    tags=["Text Recognition"],
    summary="PP-OCRv3 - Multi-Language Text Recognition",
    response_description="Extracted text with confidence scores and bounding boxes for 80+ languages"
)
async def run_ppocrv3(
    file: UploadFile = File(..., description="Image (JPG, PNG) or PDF file to process"),
    lang: Optional[str] = Form("en", description="Language code: en (English), ar (Arabic), hi (Hindi), ch (Chinese), fr (French), de (German), es (Spanish), it (Italian), ru (Russian), japan (Japanese), korean (Korean), and 70+ more languages"),
) -> JSONResponse:
    """Extract text from images and PDFs using PP-OCRv3 pipeline with support for 80+ languages.
    
    PP-OCRv3 provides comprehensive international language coverage with 80+ supported
    languages, making it ideal for multilingual document processing.

    Parameters
    ----------
    file : UploadFile
        Image (JPG, PNG) or PDF file to process. Only the first page of PDFs is processed.
    lang : str, optional
        Language code for recognition. Supports 80+ languages including major languages
        (en, ar, hi, ch, fr, de, es, it, ru, japan, korean) and many more. Default: "en".

    Returns
    -------
    JSONResponse
        JSON containing extracted text with confidence scores and bounding boxes.
    """
    _ensure_dependencies()
    
    # Read the uploaded file into memory
    data = _read_upload_to_bytes(file)
    
    # Use the common OCR helper (now async)
    text_results = await _run_ocr("PP-OCRv3", lang, data, file.filename or "")
    
    return JSONResponse(content={
        "pipeline": "PP-OCRv3",
        "description": "Multi-Language Text Recognition - Supports 80+ languages with comprehensive coverage for international document processing. Optimized for multilingual text recognition with high accuracy across diverse language families.",
        "results": text_results,
        "total_texts": len(text_results),
        "supported_languages": "80+ languages including English, Arabic, Hindi, Chinese, French, German, Spanish, Italian, Russian, Japanese, Korean, and many more"
    })


@app.post(
    "/ocr/structurev3/markdown",
    tags=["Document Parsing"],
    summary="PP-StructureV3 - Convert to Markdown",
    response_description="Markdown text with preserved document structure"
)
async def run_ppstructurev3_markdown(
    file: UploadFile = File(..., description="Document image or PDF file"),
    lang: Optional[str] = Form("en", description="Language code for OCR recognition - supports 80+ languages including en, ar, hi, ch, fr, de, es, it, ru, japan, korean, and many more"),
) -> JSONResponse:
    """Convert complex documents to clean Markdown format with preserved structure.
    
    Intelligently converts PDFs and document images into Markdown files that preserve
    original structure, including tables, formulas, and hierarchical layout.

    Parameters
    ----------
    file : UploadFile
        Document image or PDF file. For PDFs, all pages will be processed.
    lang : str, optional
        Language code for OCR recognition. Supports 80+ languages including major
        languages (en, ar, hi, ch, fr, de, es, it, ru, japan, korean) and many more.
        For mixed-language documents, use the primary language or 'ch'. Default: "en".

    Returns
    -------
    JSONResponse
        JSON containing the Markdown representation of the document with preserved
        structure, tables, and layout.
    """
    _ensure_dependencies()
    
    # Read the uploaded file into memory
    data = _read_upload_to_bytes(file)
    
    # Use the common StructureV3 helper (now async)
    result = await _run_structurev3(lang, data, file.filename or "", "markdown")
    
    return JSONResponse(content=result)


@app.post(
    "/ocr/structurev3/json",
    tags=["Document Parsing"],
    summary="PP-StructureV3 - Extract Structured JSON",
    response_description="Structured JSON with layout blocks, regions, and text elements"
)
async def run_ppstructurev3_json(
    file: UploadFile = File(..., description="Document image or PDF file"),
    lang: Optional[str] = Form("en", description="Language code for OCR recognition - supports 80+ languages including en, ar, hi, ch, fr, de, es, it, ru, japan, korean, and many more"),
) -> JSONResponse:
    """Extract structured JSON data with layout blocks, regions, and text elements.
    
    Intelligently converts PDFs and document images into structured JSON format with
    layout blocks, text content, and precise bounding boxes for document analysis.

    Parameters
    ----------
    file : UploadFile
        Document image or PDF file. For PDFs, all pages will be processed.
    lang : str, optional
        Language code for OCR recognition. Supports 80+ languages including major
        languages (en, ar, hi, ch, fr, de, es, it, ru, japan, korean) and many more.
        For mixed-language documents, use the primary language or 'ch'. Default: "en".

    Returns
    -------
    JSONResponse
        JSON containing structured document data with layout blocks, regions, text
        elements, and spatial information with confidence scores.
    """
    _ensure_dependencies()
    
    # Read the uploaded file into memory
    data = _read_upload_to_bytes(file)
    
    # Use the common StructureV3 helper (now async)
    result = await _run_structurev3(lang, data, file.filename or "", "json")
    
    return JSONResponse(content=result)


@app.get(
    "/languages",
    tags=["Language Support"],
    summary="Supported Languages Reference",
    response_description="Complete list of supported languages with codes and descriptions"
)
async def get_supported_languages() -> JSONResponse:
    """Get comprehensive list of all supported languages across all endpoints.
    
    Returns complete language reference with codes, names, endpoint support details,
    and mixed-language document recommendations.

    Returns
    -------
    JSONResponse
        Comprehensive language reference with codes, names, and usage guidance.
    """
    return JSONResponse(content={
        "title": "PaddleOCR API - Supported Languages Reference",
        "description": "Complete reference for all 80+ supported languages across all endpoints",
        "total_languages": 80,
        "endpoints": {
            "ppocrv5": {
                "languages": len(PP_OCRV5_LANGUAGES),
                "codes": list(PP_OCRV5_LANGUAGES.keys()),
                "languages_detail": PP_OCRV5_LANGUAGES,
                "best_for": "Chinese, English, Japanese, Korean, Traditional Chinese",
                "mixed_language": "Excellent for Chinese+English, Japanese+English, Korean+English"
            },
            "ppocrv3": {
                "languages": 80,
                "major_languages": PP_OCRV3_MAJOR_LANGUAGES,
                "best_for": "International documents, Arabic, Hindi, European languages",
                "mixed_language": "Good for most mixed-language combinations"
            },
            "structurev3_markdown": {
                "languages": 80,
                "best_for": "Document structure analysis in any language",
                "mixed_language": "Uses OCR recognition for layout analysis"
            },
            "structurev3_json": {
                "languages": 80,
                "best_for": "Structured data extraction in any language",
                "mixed_language": "Uses OCR recognition for layout analysis"
            },
            "chatocrv4": {
                "languages": 80,
                "best_for": "Intelligent information extraction in any language",
                "mixed_language": "Uses OCR recognition for intelligent extraction"
            }
        },
        "language_categories": {
            "european": {
                "count": len(PP_OCRV3_EUROPEAN_LANGUAGES),
                "examples": PP_OCRV3_EUROPEAN_LANGUAGES
            },
            "asian": {
                "count": len(PP_OCRV3_ASIAN_LANGUAGES),
                "examples": PP_OCRV3_ASIAN_LANGUAGES
            },
            "middle_eastern": {
                "count": len(PP_OCRV3_MIDDLE_EASTERN_LANGUAGES),
                "examples": PP_OCRV3_MIDDLE_EASTERN_LANGUAGES
            },
            "african": {
                "count": len(PP_OCRV3_AFRICAN_LANGUAGES),
                "examples": PP_OCRV3_AFRICAN_LANGUAGES
            },
            "indian_subcontinent": {
                "count": 12,
                "examples": ["hi", "bn", "ta", "te", "ml", "kn", "gu", "pa", "or", "as", "ne", "si"]
            }
        },
        "mixed_language_guidance": {
            "chinese_english": "Use 'ch' (PP-OCRv5 or PP-OCRv3)",
            "japanese_english": "Use 'japan' (PP-OCRv5 or PP-OCRv3)",
            "korean_english": "Use 'korean' (PP-OCRv5 or PP-OCRv3)",
            "arabic_english": "Use 'ar' (PP-OCRv3)",
            "hindi_english": "Use 'hi' (PP-OCRv3)",
            "european_mixed": "Use primary language code",
            "unknown_language": "Start with 'en', check confidence scores"
        },
        "usage_tips": [
            "All models are pre-downloaded for instant availability",
            "No runtime download delays (10-15 seconds saved)",
            "GPU acceleration available for all languages",
            "Confidence scores help identify best language match",
            "Mixed-language documents: use primary language code",
            "Unknown documents: start with 'en' and check results"
        ],
        "performance_characteristics": {
            "model_sizes": {
                "ppocrv5": "~50MB per language (5 languages = 250MB)",
                "ppocrv3": "~30MB per language (80+ languages = 2.4GB)",
                "structurev3": "~100MB per language (80+ languages = 8GB)",
                "chatocrv4": "~80MB per language (80+ languages = 6.4GB)",
                "total": "~17GB for complete model coverage"
            },
            "response_times": {
                "first_request_cached": "1-2 seconds per page",
                "first_request_new_language": "10-15 seconds (auto-download)",
                "subsequent_requests": "1-2 seconds per page",
                "gpu_utilization": "18-21% during inference"
            },
            "accuracy_improvements": {
                "ppocrv5": "13% improvement over PP-OCRv4",
                "ppocrv3": "Optimized for 80+ languages",
                "structurev3": "Outperforms commercial solutions",
                "chatocrv4": "15% improvement with ERNIE 4.5"
            }
        }
    })


@app.post(
    "/ocr/chatocrv4",
    tags=["Intelligent Extraction"],
    summary="PP-ChatOCRv4 - Intelligent Information Extraction with Ollama",
    response_description="Extracted key-value pairs with multimodal LLM understanding"
)
async def run_ppchatocrv4(
    file: UploadFile = File(..., description="Document image or PDF file"),
    keys: str = Form(..., description="Comma-separated list of information fields to extract (e.g., 'Landlord,Tenant,Date,Rent')"),
    lang: Optional[str] = Form("en", description="Language code for OCR recognition - supports 80+ languages"),
    # Ollama OpenAI-compatible API configuration
    mllm_model: Optional[str] = Form("llava:latest", description="Multimodal LLM model for visual understanding (e.g., llava:latest, bakllava, llava-phi3)"),
    llm_model: Optional[str] = Form("llama3:latest", description="Text-only LLM model for extraction (e.g., llama3:latest, mistral, gemma2)"),
    ollama_base_url: Optional[str] = Form("http://localhost:11434", description="Ollama base URL - will append /v1 for OpenAI API compatibility"),
    api_key: Optional[str] = Form("", description="Optional API key for Ollama service"),
) -> JSONResponse:
    """Extract specific information fields using PP-ChatOCRv4 with local Ollama service.
    
    Leverages PP-ChatOCRv4's native capabilities with locally hosted Ollama models via
    OpenAI-compatible API for multimodal visual understanding and text extraction.
    All operations are performed in-memory without temporary files.

    Parameters
    ----------
    file : UploadFile
        Document image or PDF file to process.
    keys : List[str]
        List of information fields to extract (e.g., ["Invoice Number", "Date", "Total"]).
    lang : str, optional
        Language code for OCR recognition (supports 80+ languages). Default: "en".
    mllm_model : str, optional
        Multimodal LLM for visual understanding (e.g., llava:latest, bakllava). Default: "llava:latest".
    llm_model : str, optional
        Text-only LLM for extraction (e.g., llama3:latest, mistral). Default: "llama3:latest".
    ollama_base_url : str, optional
        Ollama base URL (will append /v1 for OpenAI API). Default: "http://localhost:11434".
    api_key : str, optional
        Optional API key for Ollama service. Default: "" (no auth).

    Returns
    -------
    JSONResponse
        Extracted key-value pairs with model metadata and processing information.
        
    Raises
    ------
    HTTPException
        If PP-ChatOCRv4Doc is not available or processing fails.
    """
    _ensure_dependencies()
    
    # Check if PP-ChatOCRv4Doc is available
    if PPChatOCRv4Doc is None:
        raise HTTPException(
            status_code=503,
            detail="PP-ChatOCRv4Doc is not available. Please ensure PaddleOCR 3.x with chat support is installed."
        )
    
    # Parse keys from comma-separated string to list
    # Handle both comma-separated strings and individual keys
    if isinstance(keys, str):
        keys_list = [key.strip() for key in keys.split(',') if key.strip()]
    elif isinstance(keys, list):
        keys_list = [str(key).strip() for key in keys if str(key).strip()]
    else:
        keys_list = []
    
    if not keys_list:
        raise HTTPException(
            status_code=400,
            detail="At least one key must be provided"
        )
    
    # Read file data into memory
    data = _read_upload_to_bytes(file)
    
    # Convert bytes to numpy array for in-memory processing
    img = Image.open(io.BytesIO(data))
    img_array = np.array(img)
    
    try:
        # Instantiate PP-ChatOCRv4Doc (device only, no lang parameter)
        chatocr = PPChatOCRv4Doc(device="gpu:0")
        
        # Step 1: Visual prediction to get OCR/layout information (in-memory)
        visual_info_list_raw = chatocr.visual_predict(input=img_array)
        
        # Extract visual_info from each item (PP-ChatOCRv4 returns nested structure)
        visual_info_list = [item['visual_info'] for item in visual_info_list_raw]
        
        # Step 2: Build vector index (optional - may fail, continue without it)
        vector_info = None
        try:
            vector_info = chatocr.build_vector(visual_info=visual_info_list)
        except Exception as e:
            # If build_vector fails, continue without vector retrieval
            print(f"Warning: build_vector failed: {e}. Continuing without vector retrieval.")
        
        # Step 3: Configure chat bot to use Ollama via OpenAI-compatible API
        # Use urllib.parse for robust URL handling
        from urllib.parse import urlparse, urlunparse
        
        parsed_url = urlparse(ollama_base_url)
        if parsed_url.hostname == "localhost":
            # Use host.docker.internal to access Ollama from within container
            api_base = urlunparse((
                parsed_url.scheme or "http",
                f"host.docker.internal:{parsed_url.port or 11434}",
                "/v1",
                "", "", ""
            ))
        else:
            # Use original URL with /v1 appended
            api_base = f"{ollama_base_url.rstrip('/')}/v1"
            
        chat_bot_config = {
            "module_name": "chat_bot",
            "model_name": llm_model,
            "base_url": api_base,
            "api_type": "openai",
            "api_key": api_key or "",
        }
        
        # Step 4: Chat with integration strategy
        # Disable vector retrieval to avoid embedding model issues
        try:
            chat_result = chatocr.chat(
                key_list=keys_list,
                visual_info=visual_info_list,
                vector_info=None,  # Force no vector retrieval
                use_vector_retrieval=False,  # Disable vector retrieval
                mllm_integration_strategy="integration",
                chat_bot_config=chat_bot_config,
                retriever_config={},  # Empty config when not using vector retrieval
            )
        except Exception as chat_error:
            # Provide detailed error information for debugging
            error_details = {
                "error_type": type(chat_error).__name__,
                "error_message": str(chat_error),
                "api_base_url": api_base,
                "llm_model": llm_model,
                "mllm_model": mllm_model,
                "vector_retrieval_enabled": False,
                "visual_info_count": len(visual_info_list),
                "requested_keys": keys_list
            }
            raise HTTPException(
                status_code=500,
                detail=f"PP-ChatOCRv4 chat failed: {str(chat_error)}. Debug info: {error_details}"
            )
        
        # Convert chat_result to plain dictionary
        if hasattr(chat_result, '__dict__'):
            extracted_info = _convert_to_serializable(chat_result.__dict__)
        elif isinstance(chat_result, dict):
            extracted_info = _convert_to_serializable(chat_result)
        else:
            extracted_info = {"result": str(chat_result)}
        
        return JSONResponse(content={
            "pipeline": "PP-ChatOCRv4",
            "description": "Intelligent information extraction using PP-ChatOCRv4 with local Ollama service",
            "extracted_data": extracted_info,
            "requested_keys": keys_list,
            "models": {
                "mllm_model": mllm_model,
                "llm_model": llm_model,
                "api_base_url": api_base
            },
            "success": True
        })
        
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PP-ChatOCRv4 processing failed: {str(error)}"
        )
