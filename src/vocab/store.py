from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from src.vocab.models import UserProgress, WordLearningState

STATE_PATH = Path('.streamlit/vocab_state.json')


def load_progress() -> UserProgress:
    if not STATE_PATH.exists():
        return UserProgress()

    raw: Dict[str, Any] = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    progress = UserProgress(
        profile_name=raw.get('profile_name', 'Learner'),
        goal=raw.get('goal', 'Improve everyday communication'),
        xp=raw.get('xp', 0),
        level=raw.get('level', 1),
        streak_days=raw.get('streak_days', 0),
        longest_streak=raw.get('longest_streak', 0),
        total_words_learned=raw.get('total_words_learned', 0),
        current_day=raw.get('current_day', 1),
        completed_today=raw.get('completed_today', 0),
        daily_goal=raw.get('daily_goal', 8),
        favorites=raw.get('favorites', []),
        custom_lists=raw.get('custom_lists', {'My Core List': []}),
        forgotten_words=raw.get('forgotten_words', []),
        history=raw.get('history', []),
    )

    word_states = {}
    for word_id, state in raw.get('word_states', {}).items():
        word_states[word_id] = WordLearningState(
            word_id=word_id,
            mastery=state.get('mastery', 0.0),
            review_interval_days=state.get('review_interval_days', 1),
            ease_factor=state.get('ease_factor', 2.5),
            repetitions=state.get('repetitions', 0),
            due_day=state.get('due_day', 0),
            last_score=state.get('last_score', 0.0),
            forgotten_count=state.get('forgotten_count', 0),
            attempts=state.get('attempts', 0),
            correct_attempts=state.get('correct_attempts', 0),
        )
    progress.word_states = word_states
    return progress


def save_progress(progress: UserProgress) -> None:
    payload = asdict(progress)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
