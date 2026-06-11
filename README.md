# Narrative-Behavior Alignment Engine

A lightweight Python library that detects gaps between **what people say about themselves** (self-talk / narrative) and **what they actually do** (observable behaviors), using semantic similarity and evidence-volume confidence intervals — with zero external ML dependencies.

---

## What It Does

Given a set of behavioral events and self-referential statements for a life domain (fitness, career, finance, etc.), the engine:

1. **Computes semantic similarity** between behaviors and self-talk using TF-IDF cosine similarity (no sentence-transformers needed).
2. **Scores alignment** as a weighted blend of semantic similarity (60%) and count-ratio (40%).
3. **Classifies the gap** into one of five types.
4. **Attaches a Wilson-score confidence interval** based on evidence volume.
5. **Abstains with a clear reason** when evidence is too sparse for any reliable inference.

---

## Output Fields

Each domain in the result object contains:

| Field                  | Type    | Description                                                         |
|------------------------|---------|---------------------------------------------------------------------|
| `alignment_score`      | float   | Blended score in [0, 1]. Higher = more aligned.                     |
| `semantic_similarity`  | float   | Raw TF-IDF cosine similarity between behaviors and self-talk.       |
| `count_ratio_score`    | float   | Count-based alignment ratio (legacy method, kept for transparency). |
| `classification`       | string  | One of the five gap types below.                                    |
| `confidence.n_total`   | int     | Total evidence items (behaviors + self-talk).                       |
| `confidence.interval`  | [lo, hi]| Wilson-score 90% CI on evidence volume.                             |
| `confidence.width`     | float   | Interval width; narrower = more reliable.                           |
| `confidence.reliability` | string | `high` / `moderate` / `low` / `insufficient`.                    |

When abstaining:

| Field               | Type   | Description                               |
|---------------------|--------|-------------------------------------------|
| `classification`    | string | Always `"insufficient_evidence"`.         |
| `abstention_reason` | string | Human-readable explanation of which threshold failed. |
| `confidence`        | object | Still computed, to show how sparse the data is.       |

---

## Gap Classifications

| Type                  | Meaning                                                                          |
|-----------------------|----------------------------------------------------------------------------------|
| **aligned**           | Behaviors and self-talk describe the same pattern (semantic similarity ≥ 0.65). |
| **blind_spot**        | ≥ 3 behavioral events but zero self-talk. Person acts but doesn't narrate it.   |
| **aspiration_gap**    | ≥ 3 self-talk entries but ≤ 1 behavior. Narrative far exceeds action.            |
| **overstatement**     | Self-talk count exceeds behavior count and semantic similarity is low.           |
| **understatement**    | Behavior count exceeds self-talk count and semantic similarity is low.           |
| **insufficient_evidence** | Abstention triggered — see `abstention_reason`.                            |

---

## Abstention Thresholds

All three conditions must be met to proceed with analysis:

| Condition                        | Minimum |
|----------------------------------|---------|
| Behavioral events                | 2       |
| Self-talk entries                | 1       |
| Combined (behaviors + self-talk) | 3       |

---

## Quickstart

```python
from run import score_domains

data = {
    "fitness": {
        "behaviors": [
            "went to gym and lifted weights",
            "ran 5km in the morning",
            "attended yoga class",
        ],
        "self_talk": [
            "I exercise regularly to stay healthy",
            "working out is part of my daily routine",
        ],
    },
    "career": {
        "behaviors": ["submitted one job application"],
        "self_talk": [
            "I am building my startup every day",
            "I will become a successful founder",
            "I work on my business constantly",
        ],
    },
}

import json
print(json.dumps(score_domains(data), indent=2))
```

**Example output:**
```json
{
  "fitness": {
    "alignment_score": 0.62,
    "semantic_similarity": 0.71,
    "count_ratio_score": 0.5,
    "classification": "aligned",
    "confidence": {
      "n_total": 5,
      "interval": [0.31, 0.69],
      "width": 0.38,
      "reliability": "moderate"
    }
  },
  "career": {
    "alignment_score": 0.07,
    "semantic_similarity": 0.05,
    "count_ratio_score": 0.2,
    "classification": "aspiration_gap",
    "confidence": {
      "n_total": 4,
      "interval": [0.24, 0.76],
      "width": 0.52,
      "reliability": "low"
    }
  }
}
```

---

## Running the Tests

```bash
python -m pytest tests/test_logic.py -v
```

The test suite covers:

- `tokenize` and `meaningful_tokens` (stopword removal, punctuation, edge cases)
- `cosine_similarity` (identical, orthogonal, partial overlap, empty vectors)
- `semantic_similarity` (related vs. unrelated texts, empty inputs)
- `wilson_confidence` (ordering, range, monotonicity with n)
- `evidence_confidence` (reliability labels, n_total calculation)
- `abstain_reason` (each threshold independently, boundary values)
- `classify` (all five types, boundary counts, similarity thresholds)
- `score_domains` end-to-end (multi-domain, empty input, output structure)

---

## How Semantic Similarity Works (No ML Libraries)

```
For each domain:

1. Collect all texts (behaviors + self-talk) → build corpus-level IDF.
2. For each behavior text:
   a. Tokenize → remove stopwords → compute TF-IDF vector.
   b. Compare against every self-talk TF-IDF vector via cosine similarity.
   c. Take the maximum score (best-matching self-talk entry).
3. Average those per-behavior max scores → domain semantic_similarity.
```

This is a **bag-of-words** approach. It cannot detect pure synonyms ("gym" ≠ "workout") without a pre-trained model, but it handles shared content words well and requires no dependencies beyond Python's standard library.

---

## File Structure

```
.
├── run.py                        # Core engine (scoring, similarity, abstention)
├── decisions.md                  # Design decisions and rationale
├── requirements.txt              # pandas, numpy, pytest (numpy/pandas optional)
├── results/
│   └── example_results.json     # Pre-computed output for the worked examples
└── tests/
    └── test_logic.py            # Full pytest suite (40+ tests)
```

---

## What This System Will Never Produce

- Medical claims or diagnoses
- Psychological personality judgments
- Labels like *lazy*, *disciplined*, *intelligent*, or *unmotivated*

All outputs describe only measured narrative-behavior gaps. Interpretation is left entirely to the user.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No synonym detection | "gym" and "workout" score as dissimilar | Use longer, descriptive text |
| Short vocabulary texts | Low TF-IDF discrimination | Encourage full-sentence inputs |
| Small corpora (n < 5) | Unstable IDF weights | Check `confidence.reliability` |
| No temporal weighting | Old and new behaviors treated equally | Add timestamps if recency matters |
