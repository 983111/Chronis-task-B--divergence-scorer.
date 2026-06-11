# Decisions

## Alignment Method

Hybrid alignment: 40% count-ratio + 60% semantic similarity.

### Semantic Similarity
- Texts are tokenized and stop-words removed.
- A TF-IDF vector is built per text using a corpus-level IDF computed
  from all behaviors and self-talk in the same domain.
- Cosine similarity is computed between every behavior and every self-talk entry.
- The best-matching self-talk score is taken per behavior (max pooling).
- The mean of those per-behavior best-match scores is the domain's
  `semantic_similarity` value.

### Count-Ratio Score
`1 - abs(behavior_count - self_talk_count) / max(counts)`

Range: 0 to 1

### Final Alignment Score
`alignment = 0.4 × count_ratio_score + 0.6 × semantic_similarity`

---

## Confidence Intervals

Wilson-score confidence intervals are computed with `p = 0.5` (maximum
uncertainty) purely as a function of total evidence volume `n = n_behaviors + n_self_talk`.

This gives an honest width estimate:
- Larger evidence → narrower interval → higher reliability.
- No empirical proportion is assumed; the interval purely reflects data sparsity.

Reliability labels:
| n_total | Label        |
|---------|--------------|
| >= 10   | high         |
| 5 - 9   | moderate     |
| 2 - 4   | low          |
| < 2     | insufficient |

---

## Abstention Logic

### Type-specific exemptions (checked first)

`Blind Spot` and `Aspiration Gap` are defined by structural asymmetry.
Applying symmetric evidence requirements to them would make them unreachable
by construction — which is a logic error, not a safety feature.

| Type           | Exemption condition   | Rationale                               |
|----------------|-----------------------|-----------------------------------------|
| Blind Spot     | nb >= 3 AND ns == 0   | Zero self-talk is the signal, not a gap |
| Aspiration Gap | ns >= 3 AND nb <= 1   | Minimal behavior is the signal, not a gap |

If either exemption matches, abstention is skipped and classification proceeds.

### Symmetric thresholds (for all other types)

Only after the exemptions are checked do the symmetric thresholds apply.
All three must be satisfied to proceed with overstatement / understatement / aligned:

| Condition                     | Minimum | Reason if failed                               |
|-------------------------------|---------|------------------------------------------------|
| Behavioral events             | 2       | Cannot assess action patterns                  |
| Self-talk entries             | 1       | No narrative to compare against                |
| Combined (behaviors + talk)   | 3       | Combined evidence too sparse for any inference |

Failing any produces an `insufficient_evidence` result with an `abstention_reason`
string explaining exactly which threshold was missed.

---

## Type Boundaries

| Type                  | Condition                                          |
|-----------------------|----------------------------------------------------|
| Blind Spot            | behavior_count >= 3 AND self_talk_count == 0       |
| Aspiration Gap        | self_talk_count >= 3 AND behavior_count <= 1       |
| Aligned               | semantic_similarity >= 0.65                        |
| Overstatement         | self_talk_count > behavior_count (and not aligned) |
| Understatement        | behavior_count > self_talk_count (and not aligned) |
| Insufficient Evidence | any abstention condition triggered (after exemptions) |

---

## Failure Modes (Known Limitations)

- **Short vocabulary**: TF-IDF cannot detect synonyms ("gym" != "workout")
  without a pre-trained embedding model. Coverage improves with longer texts.
- **Short histories**: Confidence intervals widen significantly below n = 5;
  results should be interpreted cautiously.
- **Narrative style variance**: Highly terse self-talk ("I exercise") yields
  low cosine similarity even when semantically equivalent to richer behaviors.
- **IDF instability**: With very few documents (< 5 total texts), IDF weights
  are unreliable. The Wilson interval's "low/insufficient" labels flag this.
- **Aspiration Gap semantic score**: When nb <= 1, semantic_similarity() returns
  a single-behavior score. This is low-signal; the confidence interval will
  reflect the sparse evidence. The classification is based on counts, not similarity.

---

## Refusal Logic

The system never produces:
- Medical claims
- Psychological diagnoses
- Personality judgments
- Labels such as lazy, disciplined, intelligent, or unmotivated

Outputs only describe measured narrative-behavior gaps.

This is enforced structurally: `score_domains()` returns only numeric scores,
classification strings from a fixed enum, and evidence metadata. There is no
code path that produces free-text characterological statements.
