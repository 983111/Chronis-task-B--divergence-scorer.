import json
from collections import defaultdict

MIN_BEHAVIOR_EVENTS = 2

def evidence_sufficient(domain_data):
    return len(domain_data.get("behaviors", [])) >= MIN_BEHAVIOR_EVENTS

def alignment_score(behaviors, self_talk):
    if not behaviors and not self_talk:
        return 0.0

    behavior_strength = len(behaviors)
    narrative_strength = len(self_talk)

    if max(behavior_strength, narrative_strength) == 0:
        return 0.0

    return round(
        1 - abs(behavior_strength - narrative_strength)
        / max(behavior_strength, narrative_strength),
        2,
    )

def classify(domain_data):
    behaviors = domain_data.get("behaviors", [])
    self_talk = domain_data.get("self_talk", [])

    if not evidence_sufficient(domain_data):
        return "insufficient_evidence"

    b = len(behaviors)
    s = len(self_talk)

    if b >= 3 and s == 0:
        return "blind_spot"

    if s >= 3 and b <= 1:
        return "aspiration_gap"

    if s > b:
        return "overstatement"

    if b > s:
        return "understatement"

    return "aligned"

def score_domains(data):
    output = {}

    for domain, values in data.items():
        output[domain] = {
            "alignment_score": alignment_score(
                values.get("behaviors", []),
                values.get("self_talk", []),
            ),
            "classification": classify(values),
            "uncertainty": round(
                1 / max(len(values.get("behaviors", [])) +
                        len(values.get("self_talk", [])), 1),
                2,
            ),
        }

    return output

if __name__ == "__main__":
    sample = {
        "fitness": {
            "behaviors": ["gym", "gym", "run"],
            "self_talk": ["I exercise daily"]
        },
        "career": {
            "behaviors": ["job application"],
            "self_talk": ["I will become a founder",
                          "I am building constantly",
                          "I work every day"]
        },
        "reading": {
            "behaviors": ["read", "read", "read"],
            "self_talk": []
        }
    }

    print(json.dumps(score_domains(sample), indent=2))
