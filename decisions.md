# Decisions

## Alignment Method
Simple count-based alignment.

Behavior frequency is compared against self-talk frequency inside the same domain.

Alignment score:
1 - abs(behavior_count - self_talk_count) / max(counts)

Range: 0 to 1

## Failure Modes
- Ignores semantic meaning.
- Strong and weak statements are treated equally.
- Short histories increase uncertainty.

## Type Boundaries

Blind Spot:
- behavior_count >= 3
- self_talk_count == 0

Aspiration Gap:
- self_talk_count >= 3
- behavior_count <= 1

Overstatement:
- self_talk_count > behavior_count

Understatement:
- behavior_count > self_talk_count

Insufficient Evidence:
- fewer than 2 behavioral events

## Refusal Logic

The system never produces:
- medical claims
- psychological diagnoses
- personality judgments
- labels such as lazy, disciplined, intelligent, unmotivated

Outputs only describe measured narrative-behavior gaps.
