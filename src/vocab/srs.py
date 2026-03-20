from __future__ import annotations

from src.vocab.models import WordLearningState


def apply_review_result(state: WordLearningState, quality: int, today: int) -> WordLearningState:
    """
    Apply an SM-2 style spaced repetition update.

    quality is in [0, 5]. Lower values schedule sooner review; higher values
    increase interval and mastery. This keeps hard words resurfacing quickly.
    """
    quality = max(0, min(5, quality))
    state.attempts += 1
    state.last_score = quality / 5

    if quality < 3:
        state.repetitions = 0
        state.review_interval_days = 1
        state.forgotten_count += 1
        state.mastery = max(0.0, state.mastery - 0.12)
    else:
        state.correct_attempts += 1
        if state.repetitions == 0:
            state.review_interval_days = 1
        elif state.repetitions == 1:
            state.review_interval_days = 3
        else:
            state.review_interval_days = int(round(state.review_interval_days * state.ease_factor))
        state.repetitions += 1
        state.mastery = min(1.0, state.mastery + (0.08 + quality * 0.03))

    state.ease_factor = max(
        1.3,
        state.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    )
    state.due_day = today + max(1, state.review_interval_days)
    return state


def mastery_label(mastery: float) -> str:
    if mastery < 0.25:
        return "New"
    if mastery < 0.5:
        return "Learning"
    if mastery < 0.75:
        return "Almost There"
    if mastery < 0.9:
        return "Strong"
    return "Mastered"
