"""
Step 3 & 4: Region Filtering + Cropping
Fixes:
  - Deduplicate overlapping YOLO boxes (IoU-based)
  - Stricter junk text detection post-OCR
  - Better body→title assignment
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .step2_layout_detection import LayoutRegion, CONTENT_CLASSES, IGNORE_CLASSES


@dataclass
class ArticleBlock:
    regions: list[LayoutRegion] = field(default_factory=list)
    title_region: Optional[LayoutRegion] = None
    body_regions: list[LayoutRegion] = field(default_factory=list)
    bbox_px: list[int] = field(default_factory=list)
    bbox: list[float] = field(default_factory=list)  # normalized [x1,y1,x2,y2] for full-res crop
    crop: Optional[np.ndarray] = None
    title_crop: Optional[np.ndarray] = None


def compute_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] normalized boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_junk_text(text: str) -> bool:
    """
    Return True if OCR text is noise — separators, symbols, ads.
    Keeps text only if it has enough real Hindi/English word characters.
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Too short to be a real article
    if len(stripped) < 8:
        return True

    # Count actual word characters (Devanagari + Latin letters + digits)
    import re
    word_chars = re.findall(r'[\u0900-\u097Fa-zA-Z0-9]', stripped)
    total_chars = len(stripped.replace(' ', '').replace('\n', ''))

    if total_chars == 0:
        return True

    # If less than 40% of chars are real word chars → junk
    ratio = len(word_chars) / total_chars
    if ratio < 0.40:
        return True

    # Must have at least 3 actual word characters in a row (a real word)
    if not re.search(r'[\u0900-\u097Fa-zA-Z]{3,}', stripped):
        return True

    return False


class RegionFilterer:
    def __init__(self,
                 min_area: float = 0.003,
                 max_area: float = 0.85,
                 min_confidence: float = 0.30,
                 iou_threshold: float = 0.45):
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold

    def filter(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        filtered = []
        for r in regions:
            if r.class_name in IGNORE_CLASSES:
                continue
            if r.confidence < self.min_confidence:
                continue
            if r.area < self.min_area or r.area > self.max_area:
                continue
            filtered.append(r)

        # ── Deduplicate overlapping boxes (NMS-style) ─────────────────────
        filtered = self._nms(filtered)

        print(f"[Filter] {len(regions)} → {len(filtered)} regions after filter+dedup")
        from collections import Counter
        print(f"[Filter] Classes: {dict(Counter(r.class_name for r in filtered))}")
        return filtered

    def _nms(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """Remove duplicate/overlapping boxes, keep highest confidence."""
        if not regions:
            return []

        # Sort by confidence descending
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        kept = []

        for r in regions:
            suppressed = False
            for k in kept:
                if compute_iou(r.bbox, k.bbox) > self.iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(r)

        return kept

    def group_into_articles(self, regions: list[LayoutRegion],
                             img_h: int, img_w: int,
                             proximity_thresh: float = 0.20) -> list[ArticleBlock]:
        """
        Group layout regions into article blocks.
        Uses vertical proximity + horizontal overlap (no column detection).
        proximity_thresh=0.20 means body can be up to 20% of page height below its title.
        """
        blocks: list[ArticleBlock] = []

        title_regions = [r for r in regions
                         if r.class_name in ("title", "section_header")]
        body_regions  = [r for r in regions
                         if r.class_name in ("text", "list_item")]

        print(f"[Grouper] {len(title_regions)} titles + {len(body_regions)} body regions")

        # Each title starts its own block
        for t in title_regions:
            block = ArticleBlock()
            block.title_region = t
            block.regions.append(t)
            blocks.append(block)

        # Assign each body region to nearest title above it
        for body in body_regions:
            by_top  = body.bbox[1]
            by_bot  = body.bbox[3]
            by_left = body.bbox[0]
            by_right= body.bbox[2]
            by_cx   = (by_left + by_right) / 2

            best_block = None
            best_score = float("inf")

            for block in blocks:
                t = block.title_region
                if t is None:
                    continue
                ty_top  = t.bbox[1]
                ty_bot  = t.bbox[3]
                tx_left = t.bbox[0]
                tx_right= t.bbox[2]
                tx_cx   = (tx_left + tx_right) / 2

                # Title must start above body
                if ty_top >= by_bot:
                    continue

                # Check horizontal overlap between title and body
                overlap = min(tx_right, by_right) - max(tx_left, by_left)
                title_width = tx_right - tx_left
                body_width  = by_right - by_left
                min_width   = min(title_width, body_width)

                # Need at least 20% horizontal overlap
                if min_width > 0 and (overlap / min_width) < 0.20:
                    continue

                # Vertical gap between title bottom and body top
                gap = max(0.0, by_top - ty_bot)
                if gap > proximity_thresh:
                    continue

                # Score = vertical gap + small horizontal penalty
                score = gap + abs(tx_cx - by_cx) * 0.3
                if score < best_score:
                    best_score = score
                    best_block = block

            if best_block:
                best_block.body_regions.append(body)
                best_block.regions.append(body)
            else:
                # Orphan body block
                orphan = ArticleBlock()
                orphan.body_regions.append(body)
                orphan.regions.append(body)
                blocks.append(orphan)

        # Compute merged bounding boxes, drop empty blocks
        valid_blocks = []
        for block in blocks:
            if not block.regions:
                continue
            xs1 = [r.bbox_px[0] for r in block.regions]
            ys1 = [r.bbox_px[1] for r in block.regions]
            xs2 = [r.bbox_px[2] for r in block.regions]
            ys2 = [r.bbox_px[3] for r in block.regions]
            block.bbox_px = [min(xs1), min(ys1), max(xs2), max(ys2)]
            # Also store normalized bbox for full-res OCR crops
            bxs1 = [r.bbox[0] for r in block.regions]
            bys1 = [r.bbox[1] for r in block.regions]
            bxs2 = [r.bbox[2] for r in block.regions]
            bys2 = [r.bbox[3] for r in block.regions]
            block.bbox = [min(bxs1), min(bys1), max(bxs2), max(bys2)]
            valid_blocks.append(block)

        print(f"[Grouper] → {len(valid_blocks)} article blocks formed")
        return valid_blocks


class RegionCropper:
    def __init__(self, padding: int = 10):
        self.padding = padding

    def crop_blocks(self, img: np.ndarray,
                    blocks: list[ArticleBlock]) -> list[ArticleBlock]:
        h, w = img.shape[:2]
        min_px = 50

        for block in blocks:
            x1, y1, x2, y2 = block.bbox_px
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)

            if (x2 - x1) < min_px or (y2 - y1) < min_px:
                block.crop = None
                continue

            block.crop = img[y1:y2, x1:x2].copy()

            if block.title_region:
                tx1, ty1, tx2, ty2 = block.title_region.bbox_px
                tx1 = max(0, tx1 - self.padding)
                ty1 = max(0, ty1 - self.padding)
                tx2 = min(w, tx2 + self.padding)
                ty2 = min(h, ty2 + self.padding)
                if (tx2 - tx1) >= min_px and (ty2 - ty1) >= min_px:
                    block.title_crop = img[ty1:ty2, tx1:tx2].copy()

        return blocks