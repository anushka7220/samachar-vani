"""
Hindi Newspaper → Podcast Pipeline
Entry point: run this to process a newspaper image end-to-end.
"""

import argparse
from pathlib import Path
from pipeline.orchestrator import NewspaperPodcastPipeline


def main():
    parser = argparse.ArgumentParser(description="Hindi Newspaper to Podcast")
    parser.add_argument("--image", type=str, required=True, help="Path to newspaper image")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top articles to include")
    parser.add_argument("--tts-engine", type=str, default="gtts", choices=["gtts", "edge-tts"],
                        help="TTS engine to use")
    parser.add_argument("--debug", action="store_true", help="Save debug visuals")
    args = parser.parse_args()

    pipeline = NewspaperPodcastPipeline(
        output_dir=args.output_dir,
        top_n=args.top_n,
        tts_engine=args.tts_engine,
        debug=args.debug,
    )

    result = pipeline.run(args.image)

    print("\n✅ Pipeline complete!")
    print(f"   📝 Script : {result['script_path']}")
    print(f"   🎙️  Audio  : {result['audio_path']}")
    print(f"   📊 Report : {result['report_path']}")


if __name__ == "__main__":
    main()