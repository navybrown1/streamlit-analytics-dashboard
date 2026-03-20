from __future__ import annotations

import random
from typing import Dict, List

from src.vocab.models import VocabWord


QUIZ_TYPES = [
    "multiple_choice",
    "fill_blank",
    "context_choice",
    "matching",
]


def build_multiple_choice(word: VocabWord, all_words: List[VocabWord]) -> Dict[str, object]:
    distractors = [w.simple_definition for w in all_words if w.id != word.id]
    random.shuffle(distractors)
    options = [word.simple_definition] + distractors[:3]
    random.shuffle(options)
    return {
        "type": "multiple_choice",
        "prompt": f"What is the best simple meaning of '{word.word}'?",
        "options": options,
        "answer": word.simple_definition,
    }


def build_fill_blank(word: VocabWord) -> Dict[str, object]:
    return {
        "type": "fill_blank",
        "prompt": f"Complete the sentence with the correct word: {word.example_sentence.replace(word.word, '_____')}",
        "answer": word.word.lower(),
    }


def build_context_choice(word: VocabWord, all_words: List[VocabWord]) -> Dict[str, object]:
    distractors = [w.word for w in all_words if w.id != word.id]
    random.shuffle(distractors)
    options = [word.word] + distractors[:3]
    random.shuffle(options)
    return {
        "type": "context_choice",
        "prompt": f"Choose the word that best fits this context: {word.example_sentence.replace(word.word, '_____')}",
        "options": options,
        "answer": word.word,
    }


def build_matching(words: List[VocabWord]) -> Dict[str, object]:
    sampled = words[: min(4, len(words))]
    random.shuffle(sampled)
    left = [w.word for w in sampled]
    right = [w.simple_definition for w in sampled]
    random.shuffle(right)
    pairs = {w.word: w.simple_definition for w in sampled}
    return {
        "type": "matching",
        "prompt": "Match each word to the right definition.",
        "left": left,
        "right": right,
        "answer_pairs": pairs,
    }


def build_question(word: VocabWord, all_words: List[VocabWord], quiz_type: str | None = None) -> Dict[str, object]:
    qtype = quiz_type or random.choice(QUIZ_TYPES)
    if qtype == "fill_blank":
        return build_fill_blank(word)
    if qtype == "context_choice":
        return build_context_choice(word, all_words)
    if qtype == "matching":
        return build_matching(all_words)
    return build_multiple_choice(word, all_words)
