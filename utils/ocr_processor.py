"""OCR helpers for scanned PDF pages."""

import shutil
from typing import Any


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is needed but Tesseract is not installed."""


def page_needs_ocr(page: Any, text: str, min_text_chars: int = 40) -> bool:
    """Return whether a PDF page appears to be a scanned image page."""
    return len(text.strip()) < min_text_chars and bool(page.get_images(full=True))


def extract_page_text_with_ocr(
    page: Any,
    page_number: int,
    language: str = "eng",
    dpi: int = 200,
) -> str:
    """Extract text from one PDF page using PyMuPDF's Tesseract integration."""
    if shutil.which("tesseract") is None:
        raise OCRUnavailableError(
            "OCR is required for scanned page "
            f"{page_number}, but Tesseract was not found on PATH. "
            "Install Tesseract OCR and restart the application."
        )

    try:
        textpage = page.get_textpage_ocr(
            language=language,
            dpi=dpi,
            full=True,
        )
        return page.get_text("text", textpage=textpage).strip()
    except Exception as exc:
        raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc
