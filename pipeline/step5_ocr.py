"""
Step 5: OCR using EasyOCR
- Hindi + English bilingual support
- M1 MPS acceleration
"""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[OCR] EasyOCR not installed. Run: pip install easyocr")


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: list
    line_y: float = 0.0


class HindiOCR:
    def __init__(self, use_gpu: bool = True, languages: list[str] = None):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR required: pip install easyocr")

        self.languages = languages or ["hi", "en"]
        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            verbose=False,
        )
        print(f"[OCR] EasyOCR loaded | langs={self.languages} | gpu={use_gpu}")

    def read_region(self, img: np.ndarray,
                    min_confidence: float = 0.20) -> list[OCRResult]:
        """
        Run OCR on a single image crop.
        min_confidence lowered to 0.20 — Hindi text often scores lower than English.
        """
        if img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return []

        # Scale up small crops — EasyOCR struggles below ~100px height
        scale = 1.0
        if h < 100 or w < 100:
            scale = max(100 / h, 100 / w, 2.0)
            img = self._scale_up(img, scale)
            h, w = img.shape[:2]

        raw = self.reader.readtext(
            img,
            detail=1,
            paragraph=False,
            text_threshold=min_confidence,
            low_text=0.2,          # lowered: catches faint Hindi strokes
            link_threshold=0.3,    # lowered: connects broken Devanagari matras
            decoder="greedy",
            batch_size=8,
            contrast_ths=0.1,      # helps with newsprint contrast
            adjust_contrast=0.5,
        )

        results = []
        for (bbox, text, conf) in raw:
            if conf < min_confidence:
                continue
            text = text.strip()
            if not text:
                continue

            ys = [pt[1] for pt in bbox]
            y_center = sum(ys) / len(ys) / h

            results.append(OCRResult(
                text=text,
                confidence=conf,
                bbox=bbox,
                line_y=y_center,
            ))

        # Sort top-to-bottom, left-to-right
        results.sort(key=lambda r: (round(r.line_y * 10) / 10, r.bbox[0][0]))
        return results

    def _scale_up(self, img: np.ndarray, scale: float) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def results_to_text(self, results: list[OCRResult],
                         line_gap_threshold: float = 0.04) -> str:
        if not results:
            return ""

        lines: list[list[OCRResult]] = []
        current_line: list[OCRResult] = []
        last_y = -1.0

        for r in results:
            if last_y < 0 or abs(r.line_y - last_y) < line_gap_threshold:
                current_line.append(r)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [r]
            last_y = r.line_y

        if current_line:
            lines.append(current_line)

        text_lines = []
        for line in lines:
            line_text = " ".join(r.text for r in line)
            text_lines.append(line_text)

        return "\n".join(text_lines)

    def batch_read(self, crops: list[np.ndarray],
                   min_confidence: float = 0.20) -> list[str]:
        texts = []
        for i, crop in enumerate(crops):
            results = self.read_region(crop, min_confidence=min_confidence)
            text = self.results_to_text(results)
            texts.append(text)
            if (i + 1) % 5 == 0:
                print(f"[OCR] Processed {i+1}/{len(crops)} crops")
        return texts