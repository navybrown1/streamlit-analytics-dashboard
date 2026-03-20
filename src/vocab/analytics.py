from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src.vocab.models import UserProgress, VocabWord


def compute_rank(level: int) -> str:
    if level < 3:
        return "Word Rookie"
    if level < 6:
        return "Lexicon Builder"
    if level < 10:
        return "Nuance Navigator"
    return "Vocabulary Virtuoso"


def xp_to_next_level(level: int) -> int:
    return 120 + (level - 1) * 35


def compute_weak_categories(progress: UserProgress, words: List[VocabWord]) -> List[Dict[str, object]]:
    word_map = {w.id: w for w in words}
    bucket = Counter()
    for word_id, state in progress.word_states.items():
        if state.attempts < 2:
            continue
        category = word_map.get(word_id).category if word_map.get(word_id) else "unknown"
        weakness = max(0.0, 1.0 - state.mastery)
        bucket[category] += weakness

    rows = [{"category": cat, "weakness": score} for cat, score in bucket.items()]
    rows.sort(key=lambda item: item["weakness"], reverse=True)
    return rows[:5]


def consistency_heatmap(progress: UserProgress) -> List[List[int]]:
    # Build a simple 7 x 10 matrix to represent ten weeks of study activity.
    matrix = [[0 for _ in range(10)] for _ in range(7)]
    for item in progress.history[-70:]:
        day = int(item.get("day", 1))
        completed = int(item.get("completed", 0))
        row = (day - 1) % 7
        col = ((day - 1) // 7) % 10
        matrix[row][col] = min(4, completed)
    return matrix


def badges(progress: UserProgress) -> List[str]:
    unlocked = []
    if progress.streak_days >= 3:
        unlocked.append("Streak Starter")
    if progress.streak_days >= 14:
        unlocked.append("Consistency Pro")
    if progress.total_words_learned >= 20:
        unlocked.append("Word Collector")
    if progress.level >= 5:
        unlocked.append("Level Sprinter")
    if len(progress.favorites) >= 10:
        unlocked.append("Curator")
    return unlocked
