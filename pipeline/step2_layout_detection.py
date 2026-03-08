"""
Step 2: Layout Detection using YOLOv8
- Uses pre-trained PubLayNet / DocLayNet model
- Detects: title, text, figure, table, list regions
- M1 GPU (MPS) compatible
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# Layout class labels from DocLayNet / PubLayNet
LAYOUT_CLASSES = {
    0: "caption",
    1: "footnote",
    2: "formula",
    3: "list_item",
    4: "page_footer",
    5: "page_header",
    6: "picture",
    7: "section_header",
    8: "table",
    9: "text",
    10: "title",
}

# Which classes matter for news articles
CONTENT_CLASSES = {"title", "text", "section_header", "list_item"}
IGNORE_CLASSES = {"page_header", "page_footer", "caption", "formula", "picture"}


@dataclass
class LayoutRegion:
    bbox: list[float]       # [x1, y1, x2, y2] normalized 0-1
    class_id: int
    class_name: str
    confidence: float
    area: float = 0.0
    bbox_px: list[int] = field(default_factory=list)  # pixel coords


class LayoutDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        """
        model_path: path to custom .pt weights, or None to auto-download
                    recommended: 'juliozhao/DocLayNet-yolov8x' or 'PubLayNet-yolov8x'
        """
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.device = self._get_device()

        if model_path:
            self.model = YOLO(model_path)
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id="hantian/yolo-doclaynet",
                filename="yolov8x-doclaynet.pt",
            )
            self.model = YOLO(weights_path)

        print(f"[LayoutDetector] Using device: {self.device}")
        print(f"[LayoutDetector] Model type: {self.model.model.__class__.__name__}")

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detect(self, img: np.ndarray) -> list[LayoutRegion]:
        """
        img: BGR numpy array (preprocessed)
        Returns list of LayoutRegion sorted top-to-bottom, left-to-right
        """
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        regions = []
        h, w = img.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = LAYOUT_CLASSES.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                # xyxy in pixel space
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Normalized
                bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]
                area = (x2 - x1) * (y2 - y1) / (w * h)

                region = LayoutRegion(
                    bbox=bbox_norm,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    area=area,
                    bbox_px=[int(x1), int(y1), int(x2), int(y2)],
                )
                regions.append(region)

        # Sort: top-to-bottom, then left-to-right (newspaper reading order)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def visualize(self, img: np.ndarray, regions: list[LayoutRegion]) -> np.ndarray:
        """Draw bounding boxes on image (for debug)."""
        vis = img.copy()
        COLORS = {
            "title": (0, 0, 255),
            "section_header": (0, 128, 255),
            "text": (0, 200, 0),
            "table": (255, 165, 0),
            "picture": (200, 0, 200),
            "page_header": (128, 128, 128),
            "page_footer": (128, 128, 128),
        }

        for r in regions:
            x1, y1, x2, y2 = r.bbox_px
            color = COLORS.get(r.class_name, (180, 180, 180))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{r.class_name} {r.confidence:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return vis
