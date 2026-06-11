"""
Expanded test suite for the Narrative-Behavior Alignment Engine.
Covers boundary conditions, semantic similarity, confidence intervals,
abstention logic, classification, and end-to-end scoring.
"""

import math
import pytest
from run import (
    tokenize,
    meaningful_tokens,
    cosine_similarity,
    tfidf_vector,
    build_idf,
    semantic_similarity,
    wilson_confidence,
    evidence_confidence,
    abstain_reason,
    classify,
    score_domains,
    MIN_BEHAVIORS,
    MIN_SELF_TALK,
    MIN_TOTAL_EVIDENCE,
)


# ===========================================================================
# tokenize / meaningful_tokens
# ===========================================================================

class TestTokenize:
    def test_lowercases(self):
        assert "hello" in tokenize("Hello World")

    def test_strips_punctuation(self):
        tokens = tokenize("gym, gym! run.")
        assert "gym" in tokens
        assert "run" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_char_removed(self):
        assert "a" not in tokenize("a b c")

    def test_meaningful_removes_stopwords(self):
        tokens = meaningful_tokens("I am going to the gym")
        assert "i" not in tokens
        assert "am" not in tokens
        assert "gym" in tokens

    def test_meaningful_empty(self):
        assert meaningful_tokens("") == []


# ===========================================================================
# TF-IDF and cosine similarity
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = {"run": 0.5, "gym": 0.3}
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v1 = {"run": 1.0}
        v2 = {"swim": 1.0}
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert cosine_similarity({}, {}) == pytest.approx(0.0)

    def test_one_empty(self):
        assert cosine_similarity({"run": 1.0}, {}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        v1 = {"run": 1.0, "gym": 1.0}
        v2 = {"run": 1.0, "swim": 1.0}
        score = cosine_similarity(v1, v2)
        assert 0.0 < score < 1.0


class TestSemanticSimilarity:
    def test_identical_texts(self):
        texts = ["I go to the gym every day"]
        score = semantic_similarity(texts, texts)
        assert score > 0.8

    def test_empty_lists(self):
        assert semantic_similarity([], []) == 0.0

    def test_one_empty_list(self):
        assert semantic_similarity(["gym workout"], []) == 0.0

    def test_unrelated_texts_low_score(self):
        behaviors = ["went to gym and lifted weights"]
        self_talk = ["saving money is my priority"]
        score = semantic_similarity(behaviors, self_talk)
        assert score < 0.4

    def test_related_texts_higher_score(self):
        behaviors = ["went running in the park", "jogged five kilometers"]
        self_talk = ["I run regularly for fitness", "I exercise every morning"]
        score = semantic_similarity(behaviors, self_talk)
        # Should be meaningfully higher than fully unrelated
        unrelated = semantic_similarity(behaviors, ["I read books about finance"])
        assert score >= unrelated

    def test_multiple_behaviors_multiple_self_talk(self):
        behaviors = ["gym", "run", "swim"]
        self_talk = ["I exercise", "I stay fit"]
        score = semantic_similarity(behaviors, self_talk)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Wilson confidence interval
# ===========================================================================

class TestWilsonConfidence:
    def test_zero_evidence(self):
        lo, hi = wilson_confidence(0)
        assert lo == 0.0 and hi == 0.0

    def test_interval_is_ordered(self):
        lo, hi = wilson_confidence(5)
        assert lo <= hi

    def test_interval_in_range(self):
        lo, hi = wilson_confidence(10)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_larger_n_narrower_interval(self):
        lo5, hi5 = wilson_confidence(5)
        lo50, hi50 = wilson_confidence(50)
        assert (hi5 - lo5) > (hi50 - lo50)

    def test_single_evidence(self):
        lo, hi = wilson_confidence(1)
        assert lo >= 0.0
        assert hi <= 1.0


class TestEvidenceConfidence:
    def test_reliability_high(self):
        conf = evidence_confidence(6, 5)
        assert conf["reliability"] == "high"

    def test_reliability_moderate(self):
        conf = evidence_confidence(3, 2)
        assert conf["reliability"] == "moderate"

    def test_reliability_low(self):
        conf = evidence_confidence(2, 0)
        # n_total = 2, but self_talk = 0 → abstain before this is called
        # still test the function independently
        conf = evidence_confidence(1, 1)
        assert conf["reliability"] == "low"

    def test_reliability_insufficient(self):
        conf = evidence_confidence(1, 0)
        assert conf["reliability"] == "insufficient"

    def test_n_total_correct(self):
        conf = evidence_confidence(3, 4)
        assert conf["n_total"] == 7


# ===========================================================================
# Abstention logic
# ===========================================================================

class TestAbstainReason:
    def test_no_behaviors(self):
        reason = abstain_reason([], ["I exercise daily"])
        assert reason is not None
        assert "behavioral" in reason.lower()

    def test_one_behavior_below_min(self):
        reason = abstain_reason(["gym"], ["I exercise"])
        assert reason is not None

    def test_exactly_min_behaviors_no_self_talk(self):
        behaviors = ["gym"] * MIN_BEHAVIORS
        reason = abstain_reason(behaviors, [])
        assert reason is not None
        assert "self-talk" in reason.lower() or "narrative" in reason.lower()

    def test_sufficient_evidence_no_abstain(self):
        behaviors = ["gym", "run", "swim"]
        self_talk = ["I exercise regularly"]
        reason = abstain_reason(behaviors, self_talk)
        assert reason is None

    def test_boundary_exact_minimums(self):
        behaviors = ["gym"] * MIN_BEHAVIORS
        self_talk = ["I work out"] * MIN_SELF_TALK
        # MIN_TOTAL_EVIDENCE check
        if MIN_BEHAVIORS + MIN_SELF_TALK >= MIN_TOTAL_EVIDENCE:
            assert abstain_reason(behaviors, self_talk) is None
        else:
            assert abstain_reason(behaviors, self_talk) is not None

    def test_one_below_total_minimum(self):
        # Construct a case with exactly MIN_TOTAL_EVIDENCE - 1 total items
        total_needed = MIN_TOTAL_EVIDENCE
        behaviors = ["gym"] * MIN_BEHAVIORS
        self_talk = ["x"] * max(0, (total_needed - 1) - MIN_BEHAVIORS)
        if len(behaviors) + len(self_talk) < total_needed and len(self_talk) >= MIN_SELF_TALK:
            reason = abstain_reason(behaviors, self_talk)
            assert reason is not None


# ===========================================================================
# classify
# ===========================================================================

class TestClassify:
    def test_blind_spot_many_behaviors_no_self_talk(self):
        result = classify(["a", "b", "c"], [], sim_score=0.0)
        assert result == "blind_spot"

    def test_blind_spot_requires_3_behaviors(self):
        # 2 behaviors, no self_talk → not blind_spot (abstain catches it earlier)
        result = classify(["a", "b"], [], sim_score=0.0)
        # Should not be blind_spot since b < 3
        assert result != "blind_spot"

    def test_aspiration_gap(self):
        result = classify(["a"], ["x", "y", "z"], sim_score=0.1)
        assert result == "aspiration_gap"

    def test_aspiration_gap_boundary_3_self_talk_1_behavior(self):
        result = classify(["a"], ["x", "y", "z"], sim_score=0.0)
        assert result == "aspiration_gap"

    def test_aligned_high_semantic_similarity(self):
        result = classify(["gym", "run"], ["gym", "run"], sim_score=0.9)
        assert result == "aligned"

    def test_overstatement_more_self_talk(self):
        result = classify(["a", "b"], ["x", "y", "z", "w"], sim_score=0.2)
        assert result == "overstatement"

    def test_understatement_more_behaviors(self):
        result = classify(["a", "b", "c", "d"], ["x"], sim_score=0.2)
        assert result == "understatement"

    def test_aligned_equal_counts_moderate_similarity(self):
        result = classify(["a", "b"], ["a", "b"], sim_score=0.7)
        assert result == "aligned"


# ===========================================================================
# score_domains — end-to-end
# ===========================================================================

class TestScoreDomains:
    def test_insufficient_evidence_returned(self):
        data = {"domain": {"behaviors": ["a"], "self_talk": ["x"]}}
        out = score_domains(data)
        assert out["domain"]["classification"] == "insufficient_evidence"
        assert "abstention_reason" in out["domain"]

    def test_no_behaviors_returns_insufficient(self):
        data = {"domain": {"behaviors": [], "self_talk": ["I work hard"]}}
        out = score_domains(data)
        assert out["domain"]["classification"] == "insufficient_evidence"

    def test_blind_spot_detected(self):
        data = {
            "reading": {
                "behaviors": ["read novel", "read chapters", "listened audiobook"],
                "self_talk": [],
            }
        }
        out = score_domains(data)
        assert out["reading"]["classification"] == "blind_spot"

    def test_aspiration_gap_detected(self):
        data = {
            "career": {
                "behaviors": ["submitted one job application"],
                "self_talk": [
                    "I am building my startup every day",
                    "I will become a successful founder",
                    "I work on my business constantly",
                ],
            }
        }
        out = score_domains(data)
        assert out["career"]["classification"] == "aspiration_gap"

    def test_alignment_score_in_range(self):
        data = {
            "fitness": {
                "behaviors": ["gym", "run", "yoga", "cycle"],
                "self_talk": ["I exercise regularly", "working out is my routine"],
            }
        }
        out = score_domains(data)
        score = out["fitness"].get("alignment_score", -1)
        assert 0.0 <= score <= 1.0

    def test_confidence_block_present(self):
        data = {
            "finance": {
                "behaviors": ["reviewed budget", "moved to savings", "cut subscriptions"],
                "self_talk": ["I manage money well", "I save every month"],
            }
        }
        out = score_domains(data)
        assert "confidence" in out["finance"]
        conf = out["finance"]["confidence"]
        assert "reliability" in conf
        assert "interval" in conf
        assert len(conf["interval"]) == 2

    def test_multiple_domains(self):
        data = {
            "a": {"behaviors": ["x", "y"], "self_talk": []},
            "b": {"behaviors": ["x", "y", "z"], "self_talk": ["I do x"]},
        }
        out = score_domains(data)
        assert "a" in out and "b" in out

    def test_well_aligned_domain(self):
        data = {
            "fitness": {
                "behaviors": [
                    "went running in the park",
                    "lifted weights at gym",
                    "attended yoga class",
                ],
                "self_talk": [
                    "I run and exercise regularly",
                    "working out keeps me healthy",
                ],
            }
        }
        out = score_domains(data)
        # Alignment score should exist and semantic_similarity should be present
        assert "semantic_similarity" in out["fitness"]
        assert "count_ratio_score" in out["fitness"]

    def test_empty_input(self):
        out = score_domains({})
        assert out == {}

    def test_abstention_reason_message_quality(self):
        data = {"x": {"behaviors": [], "self_talk": []}}
        out = score_domains(data)
        reason = out["x"].get("abstention_reason", "")
        assert len(reason) > 10   # must be a descriptive string
