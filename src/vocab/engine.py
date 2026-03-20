from __future__ import annotations

import random
from datetime import date
from typing import Dict, List

from src.vocab.analytics import badges, compute_rank, compute_weak_categories, consistency_heatmap, xp_to_next_level
from src.vocab.data import VOCABULARY_DATA
from src.vocab.models import GOAL_TO_CATEGORIES, UserProgress, VocabWord, WordLearningState
from src.vocab.srs import apply_review_result, mastery_label


class VocabAppEngine:
    def __init__(self, progress: UserProgress) -> None:
        self.progress = progress
        self.words = [VocabWord(**row) for row in VOCABULARY_DATA]
        self.word_map: Dict[str, VocabWord] = {w.id: w for w in self.words}

        for word in self.words:
            if word.id not in self.progress.word_states:
                self.progress.word_states[word.id] = WordLearningState(word_id=word.id)

    def level_up_if_needed(self) -> None:
        while self.progress.xp >= xp_to_next_level(self.progress.level):
            self.progress.xp -= xp_to_next_level(self.progress.level)
            self.progress.level += 1

    def today_seed(self) -> int:
        # Stable daily seed gives consistent feed each day but changes tomorrow.
        return int(date.today().strftime('%Y%m%d')) + self.progress.current_day

    def due_reviews(self) -> List[VocabWord]:
        due = []
        for word in self.words:
            state = self.progress.word_states[word.id]
            if state.due_day <= self.progress.current_day:
                due.append(word)
        due.sort(key=lambda w: self.progress.word_states[w.id].mastery)
        return due

    def daily_feed(self, limit: int = 8) -> List[VocabWord]:
        preferred_categories = GOAL_TO_CATEGORIES.get(self.progress.goal, [])
        candidates = [
            w
            for w in self.words
            if w.category in preferred_categories and self.progress.word_states[w.id].mastery < 0.92
        ]
        if len(candidates) < limit:
            candidates = [w for w in self.words if self.progress.word_states[w.id].mastery < 0.92]

        rng = random.Random(self.today_seed())
        rng.shuffle(candidates)
        return candidates[:limit]

    def words_almost_known(self) -> List[VocabWord]:
        rows = [w for w in self.words if 0.55 <= self.progress.word_states[w.id].mastery < 0.8]
        rows.sort(key=lambda w: self.progress.word_states[w.id].mastery, reverse=True)
        return rows[:6]

    def record_result(self, word_id: str, quality: int, source: str) -> None:
        state = self.progress.word_states[word_id]
        before_mastery = state.mastery
        apply_review_result(state, quality=quality, today=self.progress.current_day)

        # XP scoring rewards accuracy and difficult words; penalties are implicit via low quality.
        word = self.word_map[word_id]
        difficulty_bonus = {"easy": 6, "medium": 10, "hard": 16}.get(word.difficulty, 8)
        earned = max(4, quality * 5 + difficulty_bonus)
        self.progress.xp += earned
        self.level_up_if_needed()

        if before_mastery < 0.9 and state.mastery >= 0.9:
            self.progress.total_words_learned += 1

        self.progress.completed_today += 1
        if quality < 3 and word_id not in self.progress.forgotten_words:
            self.progress.forgotten_words.append(word_id)

        self.progress.history.append(
            {
                "day": self.progress.current_day,
                "completed": self.progress.completed_today,
                "source": 1 if source == 'quiz' else 0,
            }
        )

    def start_new_day(self) -> None:
        if self.progress.completed_today > 0:
            self.progress.streak_days += 1
            self.progress.longest_streak = max(self.progress.longest_streak, self.progress.streak_days)
        else:
            self.progress.streak_days = 0

        self.progress.current_day += 1
        self.progress.completed_today = 0

    def toggle_favorite(self, word_id: str) -> None:
        if word_id in self.progress.favorites:
            self.progress.favorites.remove(word_id)
        else:
            self.progress.favorites.append(word_id)

    def mark_learned(self, word_id: str) -> None:
        state = self.progress.word_states[word_id]
        if state.mastery < 0.92:
            state.mastery = 0.92
            state.repetitions = max(3, state.repetitions)
            state.review_interval_days = max(7, state.review_interval_days)
            state.due_day = self.progress.current_day + state.review_interval_days
            self.progress.total_words_learned += 1

    def add_to_custom_list(self, list_name: str, word_id: str) -> None:
        self.progress.custom_lists.setdefault(list_name, [])
        if word_id not in self.progress.custom_lists[list_name]:
            self.progress.custom_lists[list_name].append(word_id)

    def progress_snapshot(self) -> Dict[str, object]:
        due = self.due_reviews()
        weak_areas = compute_weak_categories(self.progress, self.words)
        return {
            "rank": compute_rank(self.progress.level),
            "xp_to_next": xp_to_next_level(self.progress.level),
            "due_reviews": len(due),
            "weak_areas": weak_areas,
            "badges": badges(self.progress),
            "heatmap": consistency_heatmap(self.progress),
        }

    def search(self, query: str) -> List[VocabWord]:
        q = query.strip().lower()
        if not q:
            return []
        return [
            w
            for w in self.words
            if q in w.word.lower() or q in w.simple_definition.lower() or q in w.category.lower()
        ]

    def filtered_words(self, difficulty: str, category: str, mastery_bucket: str) -> List[VocabWord]:
        rows = self.words
        if difficulty != "all":
            rows = [w for w in rows if w.difficulty == difficulty]
        if category != "all":
            rows = [w for w in rows if w.category == category]
        if mastery_bucket != "all":
            if mastery_bucket == "new":
                rows = [w for w in rows if self.progress.word_states[w.id].mastery < 0.25]
            elif mastery_bucket == "learning":
                rows = [w for w in rows if 0.25 <= self.progress.word_states[w.id].mastery < 0.75]
            else:
                rows = [w for w in rows if self.progress.word_states[w.id].mastery >= 0.75]
        return rows

    def confused_pairs(self) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        for word in self.words:
            if not word.confused_with:
                continue
            pairs.append(
                {
                    "word": word.word,
                    "confused": word.confused_with[0],
                    "hint": f"{word.word}: {word.simple_definition}",
                }
            )
        return pairs[:8]

    def mastery_text(self, word_id: str) -> str:
        return mastery_label(self.progress.word_states[word_id].mastery)
