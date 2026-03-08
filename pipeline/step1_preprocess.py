"""
Step 1: Image Preprocessing
Key insight: We need TWO separate image pipelines:
  - YOLO layout detection: needs color BGR, resize to 1280px MAX
  - EasyOCR: needs HIGH resolution (300 DPI equivalent), clean binarization
    DO NOT resize the original down for OCR crops — use original full-res image
"""

import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, target_size: int = 1280):
        self.target_size = target_size

    def preprocess(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns: (yolo_img, original_fullres)
        yolo_img    — resized to target_size for layout detection
        original    — full resolution, used for OCR crops
        """
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Deskew on original resolution for best accuracy
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        original = self._deskew(original, gray)

        # YOLO image: resized + mild denoise
        yolo_img = self._resize(original.copy(), self.target_size)
        yolo_img = cv2.fastNlMeansDenoisingColored(
            yolo_img, None, h=4, hColor=4,
            templateWindowSize=7, searchWindowSize=15
        )

        print(f"[Preprocess] Original: {original.shape[1]}x{original.shape[0]}px "
              f"| YOLO input: {yolo_img.shape[1]}x{yolo_img.shape[0]}px")

        return yolo_img, original  # original kept full-res for OCR

    def _deskew(self, img: np.ndarray, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        if lines is None:
            return img

        angles = []
        for line in lines[:50]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -10 < angle < 10:
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.3:
            return img

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    def _resize(self, img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target / max(h, w)
        if scale >= 1.0:
            return img
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def to_ocr_ready(self, crop: np.ndarray) -> np.ndarray:
        """
        Prepare a high-res crop for EasyOCR.
        
        Critical changes vs previous version:
        1. No binary thresholding — EasyOCR's internal model works better on 
           grayscale than on hard-binarized images for Hindi Devanagari
        2. Upscale to minimum 150px height so EasyOCR can read small text
        3. CLAHE for local contrast enhancement on newsprint
        4. Light sharpening only — no aggressive kernel
        """
        if crop is None or crop.size == 0:
            return crop

        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]

        # 1. Upscale small crops — EasyOCR needs min ~100-150px height for Hindi
        min_height = 150
        if h < min_height:
            scale = min_height / h
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
            h, w = gray.shape[:2]

        # 2. CLAHE — enhances local contrast on newsprint without blowing out
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Light Gaussian blur to reduce newsprint grain before sharpening
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.5)

        # 4. Unsharp mask — sharper than a convolution kernel, more controlled
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # 5. Return as BGR (EasyOCR handles both, but BGR keeps it consistent)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def crop_for_ocr(self, original_fullres: np.ndarray,
                     bbox_norm: list[float],
                     padding_ratio: float = 0.01) -> np.ndarray:
        """
        Crop directly from the FULL RESOLUTION original image using
        normalized bbox coordinates from YOLO.
        
        This is the key quality fix — YOLO runs on 1280px image,
        but OCR crops come from the original 2000-4000px image.
        """
        h, w = original_fullres.shape[:2]
        x1_n, y1_n, x2_n, y2_n = bbox_norm

        # Add padding as fraction of image size
        pad_x = padding_ratio
        pad_y = padding_ratio

        x1 = max(0, int((x1_n - pad_x) * w))
        y1 = max(0, int((y1_n - pad_y) * h))
        x2 = min(w, int((x2_n + pad_x) * w))
        y2 = min(h, int((y2_n + pad_y) * h))

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        return original_fullres[y1:y2, x1:x2].copy()