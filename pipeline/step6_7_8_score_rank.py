"""
Steps 6, 7, 8: Headline Detection + Importance Scoring + Article Ranking
Fixes:
  - Filter junk OCR text (symbols, separators)
  - Deduplicate articles with same/similar headline
  - Better scoring for Hindi text
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


IMPORTANCE_KEYWORDS = [
    "ब्रेकिंग", "अपडेट", "ताजा", "खबर", "विशेष",
    "तत्काल", "महत्वपूर्ण", "बड़ी खबर",
    "सरकार", "मंत्री", "प्रधानमंत्री", "राष्ट्रपति",
    "संसद", "चुनाव", "नेता", "पार्टी",
    "बजट", "अर्थव्यवस्था", "बाजार", "रुपया",
    "महंगाई", "जीडीपी", "निवेश",
    "हत्या", "हादसा", "भूकंप", "बाढ़", "आग",
    "दुर्घटना", "हमला", "आतंक",
    "भारत", "देश", "राज्य", "राष्ट्रीय",
    "कोर्ट", "न्यायालय", "सुप्रीम", "सरकार",
    "corona", "covid", "war", "attack", "crisis",
]

_KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(k) for k in IMPORTANCE_KEYWORDS),
    re.IGNORECASE
)


def is_junk_text(text: str) -> bool:
    """Return True if text is OCR noise — symbols, separators, ads."""
    if not text or not text.strip():
        return True

    stripped = text.strip()

    if len(stripped) < 8:
        return True

    # Count real word characters (Devanagari + Latin + digits)
    word_chars = re.findall(r'[\u0900-\u097Fa-zA-Z0-9]', stripped)
    total_non_space = len(stripped.replace(' ', '').replace('\n', ''))

    if total_non_space == 0:
        return True

    # Less than 40% real chars = junk
    if len(word_chars) / total_non_space < 0.40:
        return True

    # Must have at least one real Hindi or English word (3+ letters)
    if not re.search(r'[\u0900-\u097Fa-zA-Z]{3,}', stripped):
        return True

    return False


def text_similarity(a: str, b: str) -> float:
    """
    Simple character-level Jaccard similarity between two strings.
    Returns 0.0 (totally different) to 1.0 (identical).
    """
    if not a or not b:
        return 0.0
    # Use word-level sets
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


@dataclass
class ScoredArticle:
    block_index: int
    title_text: str
    body_text: str
    full_text: str
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    rank: int = 0


class HeadlineDetector:
    def extract_headline(self, title_text: str, body_text: str) -> str:
        if title_text and len(title_text.strip()) > 3:
            lines = [l.strip() for l in title_text.split("\n") if l.strip()]
            # Return all non-junk lines joined (headlines can be multi-line)
            real_lines = [l for l in lines if len(l) > 3
                          and re.search(r'[\u0900-\u097Fa-zA-Z]{2,}', l)]
            if real_lines:
                return " ".join(real_lines[:2])  # max 2 lines for headline

        if body_text:
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]
            for line in lines[:4]:
                if (len(line) > 8
                        and re.search(r'[\u0900-\u097Fa-zA-Z]{3,}', line)):
                    return line
        return ""


class ImportanceScorer:
    def __init__(self,
                 w_position: float = 0.25,
                 w_size: float = 0.20,
                 w_length: float = 0.20,
                 w_keywords: float = 0.25,
                 w_headline: float = 0.10):
        self.weights = {
            "position": w_position,
            "size": w_size,
            "length": w_length,
            "keywords": w_keywords,
            "headline": w_headline,
        }

    def score(self, article: ScoredArticle,
              title_bbox_height_px: Optional[float],
              page_y_normalized: float,
              img_h: int) -> ScoredArticle:

        breakdown = {}

        # 1. Position score
        position_score = max(0.0, 1.0 - page_y_normalized * 1.5)
        breakdown["position"] = round(position_score, 3)

        # 2. Title size score
        if title_bbox_height_px and img_h > 0:
            size_score = min(1.0, title_bbox_height_px / (img_h * 0.08))
        else:
            size_score = 0.3
        breakdown["size"] = round(size_score, 3)

        # 3. Length score — combined title+body, threshold=200
        combined_len = len(article.body_text.strip()) + len(article.title_text.strip())
        length_score = min(1.0, combined_len / 200)
        breakdown["length"] = round(length_score, 3)

        # 4. Keyword score
        full_text = article.title_text + " " + article.body_text
        matches = _KEYWORD_PATTERN.findall(full_text)
        keyword_score = min(1.0, len(set(matches)) / 3.0)
        breakdown["keywords"] = round(keyword_score, 3)

        # 5. Headline presence bonus
        headline_score = 1.0 if article.title_text.strip() else 0.0
        breakdown["headline"] = headline_score

        total = sum(breakdown[k] * self.weights[k] for k in breakdown)
        article.score = round(total, 4)
        article.score_breakdown = breakdown
        return article


class ArticleRanker:
    def __init__(self, top_n: int = 3, dedup_threshold: float = 0.6):
        self.top_n = top_n
        self.dedup_threshold = dedup_threshold  # Jaccard similarity to consider duplicate

    def rank(self, articles: list[ScoredArticle]) -> list[ScoredArticle]:

        print(f"\n[Ranker] Received {len(articles)} articles:")
        for a in articles:
            print(f"  block={a.block_index} | score={a.score:.4f} "
                  f"| title='{a.title_text[:45]}' "
                  f"| body_len={len(a.body_text.strip())}")

        # ── Step 1: Remove junk OCR blocks ────────────────────────────────
        viable = []
        for a in articles:
            title_junk = is_junk_text(a.title_text)
            body_junk  = is_junk_text(a.body_text)
            combined   = (a.title_text + " " + a.body_text).strip()

            if is_junk_text(combined):
                print(f"  [Ranker] DROP block={a.block_index} — junk text: '{combined[:40]}'")
                continue
            viable.append(a)

        print(f"[Ranker] {len(viable)} viable after junk filter")

        if not viable:
            print("[Ranker] ⚠️  All articles junk — returning raw top scored")
            viable = sorted(articles, key=lambda a: a.score, reverse=True)

        # ── Step 2: Sort by score ──────────────────────────────────────────
        ranked = sorted(viable, key=lambda a: a.score, reverse=True)

        # ── Step 3: Deduplicate similar headlines ─────────────────────────
        deduped = []
        for a in ranked:
            is_dup = False
            for kept in deduped:
                sim = text_similarity(a.title_text, kept.title_text)
                if sim > self.dedup_threshold:
                    print(f"  [Ranker] DEDUP block={a.block_index} "
                          f"(sim={sim:.2f} with block={kept.block_index}): "
                          f"'{a.title_text[:40]}'")
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(a)

        print(f"[Ranker] {len(deduped)} after dedup (was {len(ranked)})")

        # Assign ranks
        for i, a in enumerate(deduped):
            a.rank = i + 1

        top = deduped[:self.top_n]
        print(f"[Ranker] Final top {len(top)}:")
        for a in top:
            print(f"  #{a.rank} score={a.score:.4f} | '{a.title_text[:60]}'")

        return top