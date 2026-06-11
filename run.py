"""
Narrative-Behavior Alignment Engine
=====================================
Uses TF-IDF-style semantic similarity (no external ML libraries needed)
instead of simple count matching. Includes confidence intervals based on
evidence volume and rigorous abstention logic.
"""

import json
import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


STOPWORDS = {
    "i", "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "my", "me", "we", "will", "do", "did",
    "have", "has", "had", "it", "its", "this", "that", "so", "just",
    "not", "no", "up", "out", "by", "as", "if", "more", "very",
}


def meaningful_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (no external libraries)
# ---------------------------------------------------------------------------

def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1) & set(v2)
    if not keys:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def build_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """Compute IDF over all token lists in the corpus."""
    N = len(corpus)
    df: Counter = Counter()
    for tokens in corpus:
        df.update(set(tokens))
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1.0
    return idf


def semantic_similarity(texts_a: List[str], texts_b: List[str]) -> float:
    """
    Mean pairwise cosine similarity between two groups of texts.
    Returns a value in [0, 1].
    """
    if not texts_a or not texts_b:
        return 0.0

    all_tokens = [meaningful_tokens(t) for t in texts_a + texts_b]
    idf = build_idf(all_tokens)

    vecs_a = [tfidf_vector(meaningful_tokens(t), idf) for t in texts_a]
    vecs_b = [tfidf_vector(meaningful_tokens(t), idf) for t in texts_b]

    scores = []
    for va in vecs_a:
        row = [cosine_similarity(va, vb) for vb in vecs_b]
        scores.append(max(row))          # best-match per behavior

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Confidence interval (Wilson score on evidence volume)
# ---------------------------------------------------------------------------

def wilson_confidence(n: int, z: float = 1.645) -> Tuple[float, float]:
    """
    Return (lower, upper) Wilson-score interval for a proportion estimated
    from n evidence items.  We use p=0.5 (maximum uncertainty) so the
    interval purely reflects evidence volume, not an empirical proportion.
    Works well as a proxy for "how certain can we be given n data points".
    """
    if n == 0:
        return (0.0, 0.0)
    p = 0.5
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, round(centre - margin, 3))
    hi = min(1.0, round(centre + margin, 3))
    return (lo, hi)


def evidence_confidence(n_behaviors: int, n_self_talk: int) -> Dict:
    n_total = n_behaviors + n_self_talk
    lo, hi = wilson_confidence(n_total)
    width = round(hi - lo, 3)
    # Reliability label
    if n_total >= 10:
        label = "high"
    elif n_total >= 5:
        label = "moderate"
    elif n_total >= 2:
        label = "low"
    else:
        label = "insufficient"
    return {
        "n_total": n_total,
        "interval": [lo, hi],
        "width": width,
        "reliability": label,
    }


# ---------------------------------------------------------------------------
# Abstention logic
# ---------------------------------------------------------------------------

MIN_BEHAVIORS = 2          # hard floor on behavioural events
MIN_SELF_TALK = 1          # at least one self-talk entry
MIN_TOTAL_EVIDENCE = 3     # combined floor


def abstain_reason(behaviors: List[str], self_talk: List[str]) -> Optional[str]:
    """
    Return a reason string if we should abstain, else None.

    Blind Spot and Aspiration Gap are structurally defined by an asymmetry
    (many behaviors / zero self-talk, or many self-talk / minimal behavior).
    These patterns are themselves the signal — they are exempt from the
    symmetric evidence requirements and checked first.

    For all other types (overstatement, understatement, aligned) we require
    both sides to have sufficient evidence before committing to a label.
    """
    nb, ns = len(behaviors), len(self_talk)

    # --- Type-specific exemptions (asymmetric by definition) ----------------

    # Blind Spot: behaviorally prominent, narratively absent.
    # Requires ≥ 3 behavioral events; zero self-talk is the defining feature.
    if nb >= 3 and ns == 0:
        return None   # valid blind_spot — do not abstain

    # Aspiration Gap: narratively prominent, behaviorally absent/flat.
    # Requires ≥ 3 self-talk entries; ≤ 1 behavior is the defining feature.
    if ns >= 3 and nb <= 1:
        return None   # valid aspiration_gap — do not abstain

    # --- Symmetric evidence requirements for all other types ----------------

    if nb < MIN_BEHAVIORS:
        return (
            f"Insufficient behavioral evidence: {nb} event(s) observed, "
            f"minimum required is {MIN_BEHAVIORS}."
        )

    if ns < MIN_SELF_TALK:
        return (
            "No self-talk recorded. Cannot compute narrative-behavior gap "
            "without at least one self-referential statement."
        )

    if nb + ns < MIN_TOTAL_EVIDENCE:
        return (
            f"Combined evidence too sparse: {nb + ns} total item(s), "
            f"minimum required is {MIN_TOTAL_EVIDENCE}."
        )

    return None   # proceed with analysis


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    behaviors: List[str],
    self_talk: List[str],
    sim_score: float,
) -> str:
    nb, ns = len(behaviors), len(self_talk)

    if nb >= 3 and ns == 0:
        return "blind_spot"

    if ns >= 3 and nb <= 1:
        return "aspiration_gap"

    # Semantic similarity informed classification
    if sim_score >= 0.65:
        return "aligned"

    if ns > nb:
        return "overstatement"

    if nb > ns:
        return "understatement"

    return "aligned"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_domains(data: Dict) -> Dict:
    output = {}

    for domain, values in data.items():
        behaviors: List[str] = values.get("behaviors", [])
        self_talk: List[str] = values.get("self_talk", [])

        reason = abstain_reason(behaviors, self_talk)
        if reason:
            output[domain] = {
                "classification": "insufficient_evidence",
                "abstention_reason": reason,
                "confidence": evidence_confidence(len(behaviors), len(self_talk)),
            }
            continue

        sim = semantic_similarity(behaviors, self_talk)
        conf = evidence_confidence(len(behaviors), len(self_talk))
        cls = classify(behaviors, self_talk, sim)

        # Alignment score: blend count ratio with semantic similarity
        nb, ns = len(behaviors), len(self_talk)
        count_score = round(
            1 - abs(nb - ns) / max(nb, ns, 1), 4
        )
        # Weighted: 40% count-ratio, 60% semantic
        alignment = round(0.4 * count_score + 0.6 * sim, 4)

        output[domain] = {
            "alignment_score": alignment,
            "semantic_similarity": sim,
            "count_ratio_score": count_score,
            "classification": cls,
            "confidence": conf,
        }

    return output


# ---------------------------------------------------------------------------
# Rich worked examples
# ---------------------------------------------------------------------------

WORKED_EXAMPLES = {
    "fitness": {
        "behaviors": [
            "went to gym and lifted weights",
            "ran 5km in the morning",
            "attended yoga class",
            "cycled to work",
        ],
        "self_talk": [
            "I exercise regularly to stay healthy",
            "working out is part of my daily routine",
        ],
    },
    "career": {
        "behaviors": [
            "submitted one job application",
        ],
        "self_talk": [
            "I am building my startup every day",
            "I will become a successful founder",
            "I work on my business constantly",
            "entrepreneurship is my calling",
        ],
    },
    "reading": {
        "behaviors": [
            "finished reading a novel",
            "read three chapters of non-fiction",
            "listened to an audiobook during commute",
        ],
        "self_talk": [],
    },
    "finance": {
        "behaviors": [
            "reviewed monthly budget",
            "transferred money to savings account",
            "cancelled unused subscriptions",
            "researched index fund options",
        ],
        "self_talk": [
            "I am disciplined with spending",
            "saving money is my top priority",
            "I track every expense carefully",
            "I invest a portion of every paycheck",
        ],
    },
    "social": {
        "behaviors": [
            "had coffee with a friend",
        ],
        "self_talk": [
            "I value my friendships deeply",
            "I make time for the people I care about",
            "I am a social person",
        ],
    },
}


if __name__ == "__main__":
    results = score_domains(WORKED_EXAMPLES)
    print(json.dumps(results, indent=2))
