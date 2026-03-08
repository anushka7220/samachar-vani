"""
Batch Processor: Run the pipeline on all newspaper images in a folder.
Supports resume, parallel-safe logging, and aggregate stats.

Usage:
  python batch_process.py --input-dir data/ --output-dir output/
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def get_image_paths(input_dir: str) -> list[Path]:
    return sorted([
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ])


def load_completed(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Batch Hindi Newspaper → Podcast")
    parser.add_argument("--input-dir", type=str, default="data", help="Folder with newspaper images")
    parser.add_argument("--output-dir", type=str, default="output", help="Output folder")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--tts-engine", type=str, default="gtts",
                        choices=["gtts", "edge-tts"],
                        help="'gtts' is more stable for batch; 'edge-tts' is higher quality")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-tts", action="store_true",
                        help="Skip TTS (useful for testing OCR/ranking pipeline)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "completed.log"
    stats_path = output_dir / "batch_stats.json"

    completed = load_completed(log_path) if args.resume else set()

    images = get_image_paths(args.input_dir)
    print(f"\n📂 Found {len(images)} images in '{args.input_dir}'")

    if args.resume and completed:
        images = [p for p in images if str(p) not in completed]
        print(f"   {len(images)} remaining after resume filter")

    if not images:
        print("Nothing to process. Exiting.")
        return

    # Import here so we can skip TTS cleanly
    from pipeline.orchestrator import NewspaperPodcastPipeline

    class PipelineWithSkipTTS(NewspaperPodcastPipeline):
        pass

    pipeline = PipelineWithSkipTTS(
        output_dir=args.output_dir,
        top_n=args.top_n,
        tts_engine=args.tts_engine,
        debug=args.debug,
    )

    if args.skip_tts:
        # Monkey-patch TTS to a no-op
        pipeline.tts.chunk_and_synthesize = lambda text, path, **kw: path
        pipeline.tts.synthesize = lambda text, path, **kw: path
        print("⏭️  TTS skipped (--skip-tts flag)")

    all_stats = []
    success, failed = 0, 0
    total_start = time.time()

    for i, img_path in enumerate(images):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(images)}] Processing: {img_path.name}")
        print(f"{'='*55}")

        try:
            img_output_dir = output_dir / img_path.stem
            img_output_dir.mkdir(exist_ok=True)
            pipeline.output_dir = img_output_dir

            result = pipeline.run(str(img_path))

            stat = {
                "image": img_path.name,
                "status": "success",
                "timings": result["timings"],
                "articles_found": len(result["top_articles"]),
                "script_path": result["script_path"],
                "audio_path": result["audio_path"],
            }
            all_stats.append(stat)
            success += 1

            # Log completed
            with open(log_path, "a") as f:
                f.write(str(img_path) + "\n")

        except Exception as e:
            print(f"❌ Failed: {img_path.name} — {e}")
            import traceback; traceback.print_exc()
            all_stats.append({
                "image": img_path.name,
                "status": "failed",
                "error": str(e),
            })
            failed += 1

    total_time = round(time.time() - total_start, 1)

    # Aggregate stats
    successful_stats = [s for s in all_stats if s["status"] == "success"]
    avg_time = (
        round(sum(s["timings"]["total"] for s in successful_stats) / len(successful_stats), 1)
        if successful_stats else 0
    )

    summary = {
        "run_at": datetime.now().isoformat(),
        "total_images": len(images),
        "success": success,
        "failed": failed,
        "total_time_seconds": total_time,
        "avg_time_per_image_seconds": avg_time,
        "results": all_stats,
    }

    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n{'='*55}")
    print(f"✅ Batch complete: {success}/{len(images)} succeeded in {total_time}s")
    print(f"   Average: {avg_time}s per image")
    print(f"   Stats saved: {stats_path}")


if __name__ == "__main__":
    main()