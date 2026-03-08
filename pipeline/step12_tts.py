"""
Step 12: Text-to-Speech

Key improvements:
- edge-tts with prosody tuning
- parallel section synthesis (much faster)
- proper silence between sections
- audio normalization
- gTTS fallback
"""

import asyncio
import os
import re
from typing import Literal


TTS_ENGINE = Literal["gtts", "edge-tts"]

VOICE_OPTIONS = {
    "female": "hi-IN-SwaraNeural",
    "male": "hi-IN-MadhurNeural",
}

EDGE_TTS_RATE = "+10%"
EDGE_TTS_PITCH = "-2Hz"
EDGE_TTS_VOLUME = "+5%"


class HindiTTS:

    def __init__(
        self,
        engine: TTS_ENGINE = "edge-tts",
        voice: str = "hi-IN-SwaraNeural",
        rate: str = EDGE_TTS_RATE,
        pitch: str = EDGE_TTS_PITCH,
        volume: str = EDGE_TTS_VOLUME,
    ):
        self.engine = engine
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

        print(f"[TTS] Engine: {engine} | Voice: {voice}")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def synthesize_script(self, script_text: str, output_path: str) -> str:

        output_path = str(output_path)

        if not output_path.endswith(".mp3"):
            output_path = output_path.rsplit(".", 1)[0] + ".mp3"

        sections = [s.strip() for s in script_text.split("\n\n") if s.strip()]

        # If script small → single synthesis
        if len(sections) == 1:
            return self._synthesize_section(sections[0], output_path)

        base = output_path.rsplit(".", 1)[0]

        # Run parallel synthesis
        section_paths = asyncio.run(
            self._synthesize_all_sections(sections, base)
        )

        merged = self._merge_with_pauses(section_paths, output_path)

        # cleanup
        for p in section_paths:
            try:
                os.remove(p)
            except:
                pass

        merged = self._normalize_audio(merged)

        return merged

    # compatibility with orchestrator
    def chunk_and_synthesize(self, text: str, output_path: str, **kwargs):
        return self.synthesize_script(text, output_path)

    # ─────────────────────────────────────────────
    # Parallel synthesis
    # ─────────────────────────────────────────────

    async def _synthesize_all_sections(self, sections, base):

        tasks = []
        paths = []

        for i, section in enumerate(sections):

            path = f"{base}_sec{i:02d}.mp3"
            paths.append(path)

            if self.engine == "edge-tts":
                tasks.append(self._edge_tts(section, path))
            else:
                # gtts is blocking → run in thread
                tasks.append(asyncio.to_thread(self._gtts, section, path))

        await asyncio.gather(*tasks)

        for i in range(len(paths)):
            print(f"[TTS] Section {i+1}/{len(paths)} done")

        return paths

    # ─────────────────────────────────────────────
    # Engines
    # ─────────────────────────────────────────────

    def _synthesize_section(self, text, output_path):

        if self.engine == "edge-tts":
            return asyncio.run(self._edge_tts(text, output_path))
        else:
            return self._gtts(text, output_path)

    async def _edge_tts(self, text, output_path):

        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )

        await communicate.save(output_path)

        return output_path

    def _gtts(self, text, output_path):

        from gtts import gTTS

        MAX_CHARS = 4000

        if len(text) <= MAX_CHARS:

            tts = gTTS(text=text, lang="hi", slow=False)
            tts.save(output_path)

            return output_path

        sentences = re.split(r'(?<=[।.!?])\s+', text)

        chunks = []
        current = ""

        for s in sentences:

            if len(current) + len(s) < MAX_CHARS:
                current += s + " "
            else:
                chunks.append(current.strip())
                current = s + " "

        if current:
            chunks.append(current.strip())

        base = output_path.rsplit(".", 1)[0]

        chunk_paths = []

        for i, chunk in enumerate(chunks):

            cp = f"{base}_chunk{i}.mp3"

            gTTS(text=chunk, lang="hi", slow=False).save(cp)

            chunk_paths.append(cp)

        merged = self._merge_with_pauses(chunk_paths, output_path, pause_ms=300)

        for p in chunk_paths:
            try:
                os.remove(p)
            except:
                pass

        return merged

    # ─────────────────────────────────────────────
    # Audio merging
    # ─────────────────────────────────────────────

    def _merge_with_pauses(self, audio_paths, output_path, pause_ms=800):

        from pydub import AudioSegment

        combined = AudioSegment.empty()

        pause = AudioSegment.silent(duration=pause_ms)

        for i, path in enumerate(audio_paths):

            if not os.path.exists(path):
                continue

            seg = AudioSegment.from_file(path)

            combined += seg

            if i < len(audio_paths) - 1:
                combined += pause

        combined.export(output_path, format="mp3")

        print(
            f"[TTS] Merged {len(audio_paths)} sections → {output_path}"
        )

        return output_path

    # ─────────────────────────────────────────────
    # Audio normalization
    # ─────────────────────────────────────────────

    def _normalize_audio(self, audio_path):

        try:
            from pydub import AudioSegment
            from pydub.effects import normalize
        except:
            return audio_path

        try:

            audio = AudioSegment.from_file(audio_path)

            normalized = normalize(audio)

            normalized = normalized.high_pass_filter(80)

            normalized.export(audio_path, format="mp3")

            print(f"[TTS] Audio normalized: {audio_path}")

        except Exception as e:
            print("[TTS] Normalization skipped:", e)

        return audio_path