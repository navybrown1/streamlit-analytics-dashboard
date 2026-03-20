from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VocabWord:
    id: str
    word: str
    pronunciation: str
    part_of_speech: str
    simple_definition: str
    advanced_definition: str
    example_sentence: str
    synonyms: List[str]
    antonyms: List[str]
    etymology: str
    difficulty: str
    memory_tip: str
    confused_with: List[str]
    category: str


@dataclass
class WordLearningState:
    word_id: str
    mastery: float = 0.0
    review_interval_days: int = 1
    ease_factor: float = 2.5
    repetitions: int = 0
    due_day: int = 0
    last_score: float = 0.0
    forgotten_count: int = 0
    attempts: int = 0
    correct_attempts: int = 0


@dataclass
class UserProgress:
    profile_name: str = "Learner"
    goal: str = "Improve everyday communication"
    xp: int = 0
    level: int = 1
    streak_days: int = 0
    longest_streak: int = 0
    total_words_learned: int = 0
    current_day: int = 1
    completed_today: int = 0
    daily_goal: int = 8
    favorites: List[str] = field(default_factory=list)
    custom_lists: Dict[str, List[str]] = field(default_factory=lambda: {"My Core List": []})
    forgotten_words: List[str] = field(default_factory=list)
    history: List[Dict[str, int]] = field(default_factory=list)
    word_states: Dict[str, WordLearningState] = field(default_factory=dict)


GOAL_TO_CATEGORIES = {
    "Improve professional vocabulary": ["professional", "leadership", "communication"],
    "Improve academic vocabulary": ["academic", "critical-thinking", "science"],
    "Prepare for SAT/GRE/GMAT style words": ["test-prep", "academic", "advanced"],
    "Improve everyday communication": ["everyday", "communication", "professional"],
}
