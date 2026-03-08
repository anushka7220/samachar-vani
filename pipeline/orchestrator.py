"""
Pipeline Orchestrator
Ties all 12 steps together into a single run() call.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import cv2

from .step1_preprocess import ImagePreprocessor
from .step2_layout_detection import LayoutDetector
from .step3_4_filter_crop import RegionFilterer, RegionCropper
from .step5_ocr import HindiOCR
from .step6_7_8_score_rank import (
    HeadlineDetector, ImportanceScorer, ArticleRanker, ScoredArticle
)
from .step9_10_11_script import (
    TextReconstructor, HindiSummarizer, PodcastScriptGenerator
)
from .step12_tts import HindiTTS
from pipeline.step10b_llm_refine import LocalSummaryRefiner


class NewspaperPodcastPipeline:

    def __init__(
        self,
        output_dir: str = "output",
        top_n: int = 3,
        tts_engine: str = "edge-tts",
        summarizer_method: str = "extractive",
        debug: bool = False
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.top_n = top_n
        self.debug = debug

        print("\n🗞️  Initializing Hindi Newspaper → Podcast Pipeline")
        print("=" * 55)

        # Core pipeline components
        self.preprocessor = ImagePreprocessor(target_size=1280)
        self.layout_detector = LayoutDetector()
        self.filterer = RegionFilterer()
        self.cropper = RegionCropper(padding=8)

        self.ocr = HindiOCR(use_gpu=True)

        self.headline_detector = HeadlineDetector()
        self.scorer = ImportanceScorer()
        self.ranker = ArticleRanker(top_n=top_n)

        self.reconstructor = TextReconstructor()
        self.summarizer = HindiSummarizer(method=summarizer_method)

        # LLM refinement
        self.refiner = LocalSummaryRefiner()

        self.script_gen = PodcastScriptGenerator()
        self.tts = HindiTTS(engine=tts_engine)

        print("✅ All components ready\n")

    def run(self, image_path: str) -> dict:

        img_path = Path(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = img_path.stem

        timings = {}
        t0 = time.time()

        # ───────────────── Step 1: Preprocess ─────────────────
        print("📐 Step 1: Preprocessing image...")
        t = time.time()

        processed_img, original_img = self.preprocessor.preprocess(image_path)

        timings["preprocess"] = round(time.time() - t, 2)

        img_h, img_w = processed_img.shape[:2]

        # ───────────────── Step 2: Layout Detection ───────────
        print("🔍 Step 2: Detecting layout with YOLOv8...")
        t = time.time()

        regions = self.layout_detector.detect(processed_img)

        timings["layout_detection"] = round(time.time() - t, 2)

        print(f"   Found {len(regions)} layout regions")

        # ───────────────── Step 3–4: Filter + Crop ────────────
        print("✂️  Steps 3-4: Filtering & cropping regions...")
        t = time.time()

        filtered_regions = self.filterer.filter(regions)

        blocks = self.filterer.group_into_articles(filtered_regions, img_h, img_w)

        blocks = self.cropper.crop_blocks(processed_img, blocks)

        timings["filter_crop"] = round(time.time() - t, 2)

        print(f"   {len(filtered_regions)} regions → {len(blocks)} article blocks")

        # ───────────────── Step 5: OCR ────────────────────────
        print(f"📖 Step 5: Running EasyOCR on {len(blocks)} blocks...")
        t = time.time()

        for i, block in enumerate(blocks):

            if block.crop is None:
                block._body_text_raw = ""
                block._title_text_raw = ""
                continue

            fullres_crop = self.preprocessor.crop_for_ocr(original_img, block.bbox)

            if fullres_crop is None:
                fullres_crop = block.crop

            ocr_img = self.preprocessor.to_ocr_ready(fullres_crop)

            body_results = self.ocr.read_region(ocr_img)

            block._body_text_raw = self.ocr.results_to_text(body_results)

            if block.title_region:

                fullres_title = self.preprocessor.crop_for_ocr(
                    original_img,
                    block.title_region.bbox
                )

                if fullres_title is not None:

                    title_ocr = self.preprocessor.to_ocr_ready(fullres_title)

                    title_results = self.ocr.read_region(title_ocr)

                    block._title_text_raw = self.ocr.results_to_text(title_results)

                else:
                    block._title_text_raw = ""

            else:
                block._title_text_raw = ""

        timings["ocr"] = round(time.time() - t, 2)

        # ───────────────── Step 6–8: Score & Rank ─────────────
        print("📊 Steps 6-8: Scoring & ranking articles...")
        t = time.time()

        scored_articles = []

        for i, block in enumerate(blocks):

            title_text = self.reconstructor.reconstruct(block._title_text_raw)

            body_text = self.reconstructor.reconstruct(block._body_text_raw)

            headline = self.headline_detector.extract_headline(
                title_text,
                body_text
            )

            article = ScoredArticle(
                block_index=i,
                title_text=headline,
                body_text=body_text,
                full_text=(headline + "\n" + body_text).strip()
            )

            title_height = None

            if block.title_region:
                ty1, ty2 = block.title_region.bbox_px[1], block.title_region.bbox_px[3]
                title_height = float(ty2 - ty1)

            page_y = block.bbox_px[1] / img_h if img_h > 0 else 0.5

            article = self.scorer.score(article, title_height, page_y, img_h)

            scored_articles.append(article)

        top_articles = self.ranker.rank(scored_articles)

        timings["score_rank"] = round(time.time() - t, 2)

        print(f"   Top {len(top_articles)} articles selected")

        # ───────────────── Step 9–10: Summarize + LLM refine ──
        print("📝 Steps 9-10: Reconstructing text & summarizing...")
        t = time.time()

        summaries = []

        for article in top_articles:

            summary = self.summarizer.summarize(
                article.body_text,
                article.title_text
            )

            # LLM refinement
            summary = self.refiner.refine(
                article.title_text,
                summary
            )

            summaries.append(summary)

        timings["summarize"] = round(time.time() - t, 2)

        # ───────────────── Step 11: Podcast Script ────────────
        print("🎙️  Step 11: Generating podcast script...")

        script = self.script_gen.generate(top_articles, summaries)

        script_path = self.output_dir / f"{stem}_{timestamp}_script.txt"

        script_path.write_text(script.full_text, encoding="utf-8")

        # ───────────────── Step 12: TTS ───────────────────────
        print("🔊 Step 12: Synthesizing audio...")
        t = time.time()

        audio_path = self.output_dir / f"{stem}_{timestamp}_podcast.mp3"

        audio_path = self.tts.chunk_and_synthesize(
            script.full_text,
            str(audio_path)
        )

        timings["tts"] = round(time.time() - t, 2)

        timings["total"] = round(time.time() - t0, 2)

        # ───────────────── Report ─────────────────────────────
        report = self._generate_report(
            image_path,
            top_articles,
            summaries,
            script,
            timings,
            timestamp
        )

        report_path = self.output_dir / f"{stem}_{timestamp}_report.json"

        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"\n⏱️  Timings: {timings}")

        return {
            "script_path": str(script_path),
            "audio_path": str(audio_path),
            "report_path": str(report_path),
            "top_articles": top_articles,
            "timings": timings
        }

    def _generate_report(
        self,
        image_path,
        top_articles,
        summaries,
        script,
        timings,
        timestamp
    ):

        return {
            "timestamp": timestamp,
            "image": str(image_path),
            "article_count": len(top_articles),
            "timings_seconds": timings,
            "articles": [
                {
                    "rank": a.rank,
                    "headline": a.title_text,
                    "score": a.score,
                    "score_breakdown": a.score_breakdown,
                    "summary": summaries[i],
                    "body_preview": a.body_text[:200]
                }
                for i, a in enumerate(top_articles)
            ],
            "script_preview": script.full_text[:500]
        }