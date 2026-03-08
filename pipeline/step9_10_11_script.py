"""
Steps 9, 10, 11: Text Reconstruction + Summarization + Podcast Script
Optimized version:
- Precompiled regex patterns
- Faster OCR noise filtering
- Reduced repeated regex scans
- Faster summarization scoring
"""

import re
import unicodedata
from dataclasses import dataclass, field
from .step6_7_8_score_rank import ScoredArticle, _KEYWORD_PATTERN


# ─────────────────────────────────────────────────────────────
# Precompiled Regex (performance improvement)
# ─────────────────────────────────────────────────────────────

HINDI_WORD = re.compile(r'[\u0900-\u097F]{3,}')
HINDI_CHAR = re.compile(r'[\u0900-\u097F]')
EN_WORD = re.compile(r'[a-zA-Z]{3,}')
LETTER_PATTERN = re.compile(r'[\u0900-\u097Fa-zA-Z]')
ENTITY_PATTERN = re.compile(r'\d|भारत|सरकार|कोर्ट|मंत्री')

_JUNK_LINE = re.compile(
    r'^[^a-zA-Z\u0900-\u097F]*$'
    r'|^\W+$'
    r'|^[\d\s\.\,\-\:\|]+$'
    r'|[|\\]{2,}'
)

# ─────────────────────────────────────────────────────────────
# OCR Correction Dictionary
# ─────────────────────────────────────────────────────────────

OCR_FIXES = {
    "कर्ने": "करने",
    "प्रितिया": "प्रतियां",
    "डजरायल": "इजरायल",
    "सठती": "सकती",
    "ठरने": "होने",
    "वनाने": "बनाने",
    "कितारबें": "किताबें",
    "नकार्द": "इनकार",
}


# ─────────────────────────────────────────────────────────────
# Sentence quality helpers
# ─────────────────────────────────────────────────────────────

def _word_error_rate(text: str) -> float:

    words = text.split()

    if not words:
        return 1.0

    broken = 0

    for w in words:

        if w.endswith('\u094D'):
            broken += 1

        elif len(re.sub(r'[^\u0900-\u097Fa-zA-Z]', '', w)) <= 1:
            broken += 1

        elif re.search(r'[\u0900-\u097F]\d|\d[\u0900-\u097F]', w):
            broken += 1

    return broken / len(words)


def _looks_like_real_sentence(text: str) -> bool:

    if not text:
        return False

    words = text.split()

    if len(words) < 4:
        return False

    if len(HINDI_WORD.findall(text)) < 2:
        return False

    short_words = [w for w in words if len(w) <= 2]

    if len(short_words) / len(words) > 0.5:
        return False

    return True


def _is_quality_sentence(text: str) -> bool:

    if not text:
        return False

    if len(text) < 12:
        return False

    if _JUNK_LINE.match(text):
        return False

    hindi_chars = HINDI_CHAR.findall(text)

    if len(hindi_chars) < 5:
        if len(EN_WORD.findall(text)) < 2:
            return False

    non_space = text.replace(' ', '')

    if not non_space:
        return False

    letters = LETTER_PATTERN.findall(non_space)

    if len(letters) / len(non_space) < 0.5:
        return False

    if _word_error_rate(text) > 0.30:
        return False

    return True


def _quality_score(sentence: str) -> float:

    if not sentence:
        return 0.0

    non_space = sentence.replace(' ', '')

    letters = len(LETTER_PATTERN.findall(non_space))

    letter_ratio = letters / max(len(non_space), 1)

    wer = _word_error_rate(sentence)

    length_ok = 1.0 if 20 <= len(sentence) <= 180 else 0.6

    return (letter_ratio * (1.0 - wer)) * length_ok


# ─────────────────────────────────────────────────────────────
# Step 9: Text Reconstruction
# ─────────────────────────────────────────────────────────────

class TextReconstructor:

    def reconstruct(self, raw_text: str) -> str:

        if not raw_text:
            return ""

        text = unicodedata.normalize("NFC", raw_text)

        text = re.sub(r'।।+', '।', text)
        text = re.sub(r'\s+।', '।', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        text = re.sub(r'[`~\^]', '', text)
        text = re.sub(r'\s\|\s', ' ', text)

        for wrong, correct in OCR_FIXES.items():
            text = text.replace(wrong, correct)

        lines = text.split('\n')

        clean_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if _is_quality_sentence(line) or _looks_like_real_sentence(line):
                clean_lines.append(line)

        return '\n'.join(clean_lines)


# ─────────────────────────────────────────────────────────────
# Step 10: Summarization
# ─────────────────────────────────────────────────────────────

class HindiSummarizer:

    def __init__(self, method="extractive", max_sentences=3):

        self.method = method
        self.max_sentences = max_sentences

    def summarize(self, body_text: str, title_text: str = "") -> str:

        body = body_text.strip()

        if not body:
            return ""

        return self._extractive_summarize(body, title_text)

    def _extractive_summarize(self, body: str, title: str):

        sentences = re.split(r'[।!?]+|\n{2,}', body)

        sentences = [s.strip() for s in sentences if s.strip()]

        title_clean = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', '', title).lower()

        title_tokens = set(title_clean.split())

        filtered = []

        for s in sentences:

            s_clean = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', '', s).lower()

            if title_tokens:

                sent_tokens = set(s_clean.split())

                jaccard = len(sent_tokens & title_tokens) / max(len(sent_tokens | title_tokens), 1)

                if jaccard > 0.55:
                    continue

            if not (_is_quality_sentence(s) and _looks_like_real_sentence(s)):
                continue

            filtered.append(s)

        if not filtered:
            return ""

        scored = []

        for i, sent in enumerate(filtered):

            words = sent.split()

            word_set = set(words)

            pos = 1.4 if i == 0 else (1.2 if i <= 2 else 1.0)

            kw = min(1.0, len(set(_KEYWORD_PATTERN.findall(sent))) / 2.0)

            density = len(word_set) / max(len(words), 1)

            entity_bonus = 0.2 if ENTITY_PATTERN.search(sent) else 0

            qual = _quality_score(sent)

            importance = pos * (
                0.30 * qual +
                0.25 * kw +
                0.20 * density +
                entity_bonus
            )

            scored.append((i, sent, importance))

        ranked = sorted(scored, key=lambda x: x[2], reverse=True)

        top = sorted(ranked[:self.max_sentences], key=lambda x: x[0])

        summary = '। '.join(s.rstrip('।') for _, s, _ in top) + '।'

        return summary


# ─────────────────────────────────────────────────────────────
# Step 11: Podcast Script Generator
# ─────────────────────────────────────────────────────────────

def _clean_for_speech(text: str):

    text = re.sub(r'[`~\^*#@$%|\\]', '', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'।(?!\s)', '। ', text)
    text = re.sub(r'\s*—\s*', ', ', text)
    text = re.sub(r'\s*-\s*', ' ', text)
    text = re.sub(r'  +', ' ', text)

    return text.strip()


INTRO = (
    "नमस्ते। आप सुन रहे हैं आज की प्रमुख खबरें। "
    "आज हम आपके लिए लाए हैं {count} महत्वपूर्ण समाचार। "
    "तो चलिए शुरू करते हैं।"
)

STORY_OPENER = ["पहली खबर।", "दूसरी खबर।", "तीसरी खबर।"]

TRANSITIONS = [
    "अब बात करते हैं अगली खबर की।",
    "इसके बाद एक और अहम समाचार।",
]

OUTRO = (
    "यह थीं आज की प्रमुख खबरें। "
    "उम्मीद है यह जानकारी आपके लिए उपयोगी रही। "
    "सुनते रहिए। धन्यवाद।"
)


@dataclass
class PodcastScript:

    full_text: str
    sections: list[dict] = field(default_factory=list)
    article_count: int = 0


class PodcastScriptGenerator:

    def __init__(self, include_transitions=True):

        self.include_transitions = include_transitions

    def generate(self, articles: list[ScoredArticle], summaries: list[str]):

        sections = []

        script_parts = [_clean_for_speech(INTRO.format(count=len(articles)))]

        for i, (article, summary) in enumerate(zip(articles, summaries)):

            headline = _clean_for_speech(article.title_text.strip())

            summary = _clean_for_speech(summary.strip())

            opener = STORY_OPENER[i] if i < len(STORY_OPENER) else f"{i+1}वीं खबर।"

            section_text = f"{opener} {headline}। {summary}"

            sections.append({
                "rank": i + 1,
                "headline": headline,
                "summary": summary,
                "section_text": section_text,
            })

            script_parts.append(section_text)

            if self.include_transitions and i < len(articles) - 1:
                script_parts.append(TRANSITIONS[i % len(TRANSITIONS)])

        script_parts.append(_clean_for_speech(OUTRO))

        full_text = "\n\n".join(script_parts)

        return PodcastScript(
            full_text=full_text,
            sections=sections,
            article_count=len(articles),
        )