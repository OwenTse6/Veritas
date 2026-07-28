"""
text_signals.py

Heuristic, explainable signals for detecting likely AI-generated text.
No ML models, no API calls -- just statistics anyone can inspect and tune.

Each function returns a small dict:
    {
        "score": float (0-100, higher = more "AI-like"),
        "detail": dict with the raw numbers that produced the score
    }

This makes every signal auditable -- you can always answer
"why did this get flagged?" by looking at `detail`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Shared text utilities
# --------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Good enough for statistical signals --
    we don't need perfect NLP-grade splitting, just consistency."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Signal 1: Burstiness (sentence-length variance)
# --------------------------------------------------------------------------
#
# Human writing naturally varies sentence length a lot -- short punchy
# sentences next to long winding ones. LLM output tends to regress toward
# a "comfortable" medium length with unusually low variance.
#
# We measure burstiness as the coefficient of variation (stdev / mean)
# of sentence lengths (in words). Lower CV => more suspicious.

def burstiness(text: str) -> dict:
    sentences = split_sentences(text)
    lengths = [len(tokenize(s)) for s in sentences if tokenize(s)]

    if len(lengths) < 3:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "not enough sentences", "n": len(lengths)},
        }

    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    stdev = math.sqrt(variance)
    cv = stdev / mean_len if mean_len else 0.0

    # Empirically, human prose usually has CV > 0.5; smooth LLM output
    # often sits around 0.25-0.45. These thresholds are a starting point --
    # calibrate them against your own text samples.
    if cv >= 0.6:
        score = 5.0
    elif cv <= 0.2:
        score = 90.0
    else:
        # linear interpolation between 0.2 (score 90) and 0.6 (score 5)
        score = 90.0 - ((cv - 0.2) / 0.4) * 85.0

    return {
        "score": round(score, 1),
        "detail": {
            "sentence_count": len(lengths),
            "mean_length": round(mean_len, 2),
            "stdev": round(stdev, 2),
            "coefficient_of_variation": round(cv, 3),
        },
    }


# --------------------------------------------------------------------------
# Signal 2: Vocabulary fingerprint ("AI tells")
# --------------------------------------------------------------------------
#
# LLMs disproportionately favor certain words/phrases. We score hits
# against a curated, weighted list. This is the most "explainable" signal --
# you can literally show the user the exact phrases that fired.

@dataclass
class TellList:
    phrases: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "TellList":
        phrases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                phrase, _, weight = line.rpartition("|")
                try:
                    phrases.append((phrase.lower(), float(weight)))
                except ValueError:
                    continue
        # longest phrases first so multi-word phrases match before
        # any accidental substring overlap
        phrases.sort(key=lambda p: -len(p[0]))
        return cls(phrases=phrases)


def vocab_fingerprint(text: str, tells: TellList) -> dict:
    lowered = text.lower()
    word_count = max(len(tokenize(text)), 1)

    hits: list[tuple[str, int, float]] = []  # phrase, count, weight
    weighted_total = 0.0

    for phrase, weight in tells.phrases:
        count = lowered.count(phrase)
        if count:
            hits.append((phrase, count, weight))
            weighted_total += count * weight

    # Normalize per 1000 words so short and long texts are comparable.
    per_1000 = weighted_total / word_count * 1000

    # Calibration knobs: tune against real samples.
    score = min(per_1000 * 10, 100.0)

    hits.sort(key=lambda h: -(h[1] * h[2]))

    return {
        "score": round(score, 1),
        "detail": {
            "word_count": word_count,
            "weighted_hits_per_1000_words": round(per_1000, 2),
            "top_hits": [{"phrase": p, "count": c, "weight": w} for p, c, w in hits[:10]],
        },
    }


# --------------------------------------------------------------------------
# Signal 3: Structural repetition (repeated syntactic templates)
# --------------------------------------------------------------------------
#
# LLMs love certain sentence-level templates: "Not only X, but also Y",
# "From X to Y", "Whether X or Y". We look for repeated POS-agnostic
# skeletons by stripping content words down to a lightweight template,
# and also do plain n-gram repetition on word sequences.

_TEMPLATE_PATTERNS = [
    (re.compile(r"\bnot only\b.{0,40}\bbut also\b", re.I), "not_only_but_also"),
    (re.compile(r"\bwhether\b.{0,30}\bor\b", re.I), "whether_or"),
    (re.compile(r"\bfrom\b.{0,20}\bto\b", re.I), "from_to"),
    (re.compile(r"\bit'?s not (just|only)\b.{0,40}\b(it'?s|but)\b", re.I), "not_just_its"),
]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def structural_repetition(text: str) -> dict:
    sentences = split_sentences(text)
    tokens = tokenize(text)

    template_hits = Counter()
    for sentence in sentences:
        for pattern, name in _TEMPLATE_PATTERNS:
            if pattern.search(sentence):
                template_hits[name] += 1

    # 4-gram repetition as a generic "repeated structure" proxy.
    fourgrams = _ngrams(tokens, 4)
    fourgram_counts = Counter(fourgrams)
    repeated_fourgrams = {g: c for g, c in fourgram_counts.items() if c > 1}
    repeated_ratio = sum(repeated_fourgrams.values()) / max(len(fourgrams), 1)

    sentence_count = max(len(sentences), 1)
    template_rate = sum(template_hits.values()) / sentence_count

    # Combine both signals into one score.
    score = min((template_rate * 60) + (repeated_ratio * 200), 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "template_hits": dict(template_hits),
            "template_rate_per_sentence": round(template_rate, 3),
            "repeated_fourgram_ratio": round(repeated_ratio, 3),
            "example_repeated_fourgrams": [
                " ".join(g) for g in list(repeated_fourgrams.keys())[:5]
            ],
        },
    }


# --------------------------------------------------------------------------
# Signal 4: Punctuation signature
# --------------------------------------------------------------------------
#
# AI writing has some well-known punctuation habits: heavy em-dash use,
# and "rule of three" lists (X, Y, and Z) used far more often than
# typical human prose.

_EM_DASH_RE = re.compile(r"[\u2014\u2013]")  # em dash, en dash
_RULE_OF_THREE_RE = re.compile(
    r"\b\w+,\s*\w+,?\s*(?:and|or)\s*\w+\b", re.I
)


def punctuation_signature(text: str) -> dict:
    word_count = max(len(tokenize(text)), 1)
    em_dash_count = len(_EM_DASH_RE.findall(text))
    rule_of_three_count = len(_RULE_OF_THREE_RE.findall(text))

    em_dash_per_1000 = em_dash_count / word_count * 1000
    triplet_per_1000 = rule_of_three_count / word_count * 1000

    # Calibration knobs.
    score = min(em_dash_per_1000 * 8 + triplet_per_1000 * 4, 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "em_dash_count": em_dash_count,
            "em_dash_per_1000_words": round(em_dash_per_1000, 2),
            "rule_of_three_count": rule_of_three_count,
            "rule_of_three_per_1000_words": round(triplet_per_1000, 2),
        },
    }


# --------------------------------------------------------------------------
# Signal 5: Unqualified praise (evaluative language vs. hedging)
# --------------------------------------------------------------------------
#
# Both AI and human writers use superlatives ("brilliant", "flawless").
# The difference is usually in what SURROUNDS them. Human critics -- even
# very enthusiastic ones -- tend to qualify praise somewhere ("mostly",
# "if a bit predictable", "not without flaws"). AI-generated critical
# writing (reviews, essays, analyses) often stacks unhedged superlatives
# with little or no qualification, especially in pieces that are
# nominally "critiques" but contain almost no real criticism.
#
# So instead of scoring superlative words alone (too blunt -- flags any
# genuinely enthusiastic human writer), we score the RATIO of superlative
# density to hedge density. High praise + low hedging is the actual tell;
# high praise + healthy hedging is just... a positive review.

_SUPERLATIVES = [
    "brilliant", "brilliantly", "flawless", "flawlessly", "masterful",
    "masterclass", "perfect", "perfectly", "excellent", "exceptional",
    "outstanding", "impeccable", "unparalleled", "phenomenal",
    "extraordinary", "sublime", "superb", "stellar", "remarkable",
    "underscores", "encapsulates", "resonates", "weaponized",
    "elevates", "elevated", "exemplifies",
]

_HEDGES = [
    "mostly", "somewhat", "a bit", "a little", "although", "though",
    "however", "but", "if a bit", "not without", "arguably",
    "admittedly", "granted", "to be fair", "on the other hand",
    "that said", "still,", "nonetheless", "occasionally", "at times",
    "one flaw", "one criticism", "one weakness", "falls short",
    "could have", "might have benefited", "the only",
]


def evaluative_language(text: str) -> dict:
    lowered = text.lower()
    word_count = max(len(tokenize(text)), 1)

    superlative_hits = {w: lowered.count(w) for w in _SUPERLATIVES if lowered.count(w)}
    hedge_hits = {w: lowered.count(w) for w in _HEDGES if lowered.count(w)}

    superlative_count = sum(superlative_hits.values())
    hedge_count = sum(hedge_hits.values())

    superlative_per_1000 = superlative_count / word_count * 1000
    hedge_per_1000 = hedge_count / word_count * 1000

    # Ratio-based score: praise with little/no hedging is the suspicious case.
    # +1 smoothing avoids divide-by-zero and avoids overreacting to a single
    # superlative in an otherwise hedge-free short text.
    ratio = superlative_per_1000 / (hedge_per_1000 + 1.0)

    # Needs almost no superlatives -> score stays low regardless of ratio.
    if superlative_count == 0:
        score = 0.0
    else:
        score = min(ratio * 25, 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "superlative_count": superlative_count,
            "hedge_count": hedge_count,
            "superlative_per_1000_words": round(superlative_per_1000, 2),
            "hedge_per_1000_words": round(hedge_per_1000, 2),
            "praise_to_hedge_ratio": round(ratio, 2),
            "superlatives_found": superlative_hits,
            "hedges_found": hedge_hits,
        },
    }


# --------------------------------------------------------------------------
# Convenience: run all text signals at once
# --------------------------------------------------------------------------

DEFAULT_TELLS_PATH = Path(__file__).parent.parent / "data" / "ai_tells.txt"


def analyze_text(text: str, tells_path: str | Path = DEFAULT_TELLS_PATH) -> dict:
    tells = TellList.from_file(tells_path)
    return {
        "burstiness": burstiness(text),
        "vocab_fingerprint": vocab_fingerprint(text, tells),
        "structural_repetition": structural_repetition(text),
        "punctuation_signature": punctuation_signature(text),
        "evaluative_language": evaluative_language(text),
    }
