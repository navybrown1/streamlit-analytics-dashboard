from __future__ import annotations

import random
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.vocab.engine import VocabAppEngine
from src.vocab.models import GOAL_TO_CATEGORIES
from src.vocab.quizzes import build_question
from src.vocab.store import load_progress, save_progress


st.set_page_config(page_title="LexiLift", page_icon="📚", layout="wide")


def inject_theme(dark_mode: bool) -> None:
    if dark_mode:
        bg = "#0f1720"
        panel = "#14202c"
        text = "#eaf2f7"
        soft = "#9eb4c5"
    else:
        bg = "#f3f6f8"
        panel = "#ffffff"
        text = "#0f202d"
        soft = "#5f7282"

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

            .stApp {{
                font-family: 'Manrope', sans-serif;
                background: radial-gradient(circle at 5% 10%, #8fe7c8 0%, {bg} 42%);
                color: {text};
            }}
            h1, h2, h3 {{
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.02em;
            }}
            .glass-card {{
                background: {panel};
                border-radius: 18px;
                border: 1px solid rgba(120, 145, 162, 0.22);
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 35px rgba(20, 35, 45, 0.08);
                margin-bottom: 0.75rem;
                animation: riseIn 260ms ease-out;
            }}
            .meta-text {{ color: {soft}; font-size: 0.92rem; }}
            .pill {{
                display: inline-block;
                border-radius: 999px;
                padding: 0.15rem 0.55rem;
                font-size: 0.78rem;
                margin-right: 0.32rem;
                margin-bottom: 0.2rem;
                background: rgba(59, 160, 126, 0.14);
                color: #1e7f61;
            }}
            .stButton > button {{
                border-radius: 12px;
                border: 1px solid rgba(120,145,162,0.25);
                padding: 0.45rem 0.8rem;
            }}
            @keyframes riseIn {{
                from {{ transform: translateY(8px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session() -> None:
    if "progress" not in st.session_state:
        st.session_state.progress = load_progress()
    if "selected_word" not in st.session_state:
        st.session_state.selected_word = ""
    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = None
    if "quiz_word_id" not in st.session_state:
        st.session_state.quiz_word_id = ""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def mastery_color(mastery: float) -> str:
    if mastery < 0.3:
        return "#ef4444"
    if mastery < 0.7:
        return "#f59e0b"
    return "#10b981"


def render_word_card(engine: VocabAppEngine, word_id: str, include_actions: bool = True) -> None:
    word = engine.word_map[word_id]
    state = engine.progress.word_states[word_id]
    c = mastery_color(state.mastery)

    st.markdown(
        f"""
        <div class='glass-card'>
            <h4 style='margin:0'>{word.word} <span class='meta-text'>/ {word.pronunciation} /</span></h4>
            <p class='meta-text' style='margin:0.2rem 0'>{word.part_of_speech} • {word.category} • {word.difficulty}</p>
            <p style='margin-top:0.35rem'><b>{word.simple_definition}</b></p>
            <p class='meta-text' style='margin-top:-0.4rem'>Mastery: <span style='color:{c}'>{engine.mastery_text(word_id)} ({state.mastery:.0%})</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if include_actions:
        cols = st.columns([1, 1, 1])
        if cols[0].button("Open details", key=f"detail_{word_id}"):
            st.session_state.selected_word = word_id
        if cols[1].button("Mark learned", key=f"learned_{word_id}"):
            engine.mark_learned(word_id)
            save_progress(engine.progress)
            st.success(f"{word.word} marked as learned.")
        if cols[2].button("Favorite", key=f"fav_{word_id}"):
            engine.toggle_favorite(word_id)
            save_progress(engine.progress)


def screen_onboarding(engine: VocabAppEngine) -> None:
    st.title("Welcome to LexiLift")
    st.caption("A premium daily vocabulary trainer built for active recall and long-term retention.")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Personalize your learning path")
        name = st.text_input("Display name", value=engine.progress.profile_name)
        goal = st.selectbox("Learning goal", list(GOAL_TO_CATEGORIES.keys()), index=list(GOAL_TO_CATEGORIES.keys()).index(engine.progress.goal))
        daily_goal = st.slider("Daily word target", min_value=5, max_value=20, value=engine.progress.daily_goal)

        if st.button("Start Learning", type="primary"):
            engine.progress.profile_name = name.strip() or "Learner"
            engine.progress.goal = goal
            engine.progress.daily_goal = daily_goal
            save_progress(engine.progress)
            st.success("Onboarding completed. Visit Home Dashboard.")

    with right:
        st.markdown("### What you get")
        st.markdown("- Daily feed curated to your goal")
        st.markdown("- Spaced repetition with adaptive intervals")
        st.markdown("- XP, streaks, badges, and weekly challenge")
        st.markdown("- Quiz modes: multiple-choice, matching, fill blank, context")
        st.markdown("- Progress analytics, weak area detection, and review planner")


def screen_home(engine: VocabAppEngine) -> None:
    snap = engine.progress_snapshot()
    st.title("Home Dashboard")

    a, b, c, d, e = st.columns(5)
    a.metric("Level", f"{engine.progress.level}")
    b.metric("Rank", snap["rank"])
    c.metric("XP", f"{engine.progress.xp}/{snap['xp_to_next']}")
    d.metric("Streak", f"{engine.progress.streak_days} days")
    e.metric("Due Reviews", snap["due_reviews"])

    x = min(1.0, engine.progress.completed_today / max(1, engine.progress.daily_goal))
    st.progress(x, text=f"Daily goal progress: {engine.progress.completed_today}/{engine.progress.daily_goal}")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("### Today challenge")
        due = engine.due_reviews()[:4]
        if due:
            st.info(f"Complete {len(due)} due reviews and one quiz streak of 5 to unlock +120 bonus XP.")
        else:
            st.success("No due reviews. Push mastery with fresh daily words.")

        st.markdown("### Words you almost know")
        for word in engine.words_almost_known():
            state = engine.progress.word_states[word.id]
            st.write(f"{word.word} • {state.mastery:.0%} mastery • next review day {state.due_day}")

    with c2:
        st.markdown("### Achievements")
        for badge in snap["badges"] or ["No badges yet. Keep learning daily."]:
            st.markdown(f"- {badge}")
        st.markdown("### Reminder")
        st.caption("Notification placeholder: Daily reminder at 8:00 PM local time.")
        if st.button("Advance to next day"):
            engine.start_new_day()
            save_progress(engine.progress)
            st.experimental_rerun()


def screen_daily_words(engine: VocabAppEngine) -> None:
    st.title("Daily Words")
    st.caption("Active recall feed with reveal-first interactions.")

    feed = engine.daily_feed(limit=engine.progress.daily_goal)
    if not feed:
        st.info("All words are mastered. Great work.")
        return

    for idx, word in enumerate(feed):
        st.markdown(f"### {idx + 1}. {word.word}")
        show_def = st.toggle("Tap to reveal definition", key=f"reveal_{word.id}")
        if show_def:
            render_word_card(engine, word.id)
        else:
            st.markdown(
                "<div class='glass-card'><p class='meta-text'>Try to recall the meaning before revealing.</p></div>",
                unsafe_allow_html=True,
            )

        usage = st.text_input(f"Use '{word.word}' in a sentence", key=f"usage_{word.id}")
        if st.button("Submit usage", key=f"submit_usage_{word.id}"):
            score = 4 if word.word.lower() in usage.lower() else 2
            engine.record_result(word.id, quality=score, source="usage")
            save_progress(engine.progress)
            if score >= 4:
                st.success("Great context usage. Memory reinforced.")
            else:
                st.warning("Try including the target word naturally in your sentence.")


def screen_word_detail(engine: VocabAppEngine) -> None:
    st.title("Word Detail")

    options = [w.id for w in engine.words]
    labels = {w.id: w.word for w in engine.words}
    current = st.session_state.selected_word or options[0]
    chosen = st.selectbox("Select word", options=options, index=options.index(current), format_func=lambda x: labels[x])
    st.session_state.selected_word = chosen

    word = engine.word_map[chosen]
    state = engine.progress.word_states[chosen]

    st.markdown(f"## {word.word}  ")
    st.markdown(f"**Pronunciation:** {word.pronunciation}")
    st.markdown(f"**Part of speech:** {word.part_of_speech}")
    st.markdown(f"**Simple definition:** {word.simple_definition}")
    st.markdown(f"**Advanced definition:** {word.advanced_definition}")
    st.markdown(f"**Example:** {word.example_sentence}")
    st.markdown(f"**Synonyms:** {', '.join(word.synonyms)}")
    st.markdown(f"**Antonyms:** {', '.join(word.antonyms)}")
    st.markdown(f"**Etymology:** {word.etymology}")
    st.markdown(f"**Difficulty:** {word.difficulty}")
    st.markdown(f"**Memory tip:** {word.memory_tip}")
    st.markdown(f"**Commonly confused with:** {', '.join(word.confused_with)}")
    st.markdown(f"**Mastery:** {state.mastery:.0%} ({engine.mastery_text(chosen)})")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Save favorite"):
        engine.toggle_favorite(chosen)
        save_progress(engine.progress)
    if c2.button("Mark as learned"):
        engine.mark_learned(chosen)
        save_progress(engine.progress)
    list_name = c3.text_input("Custom list name", value="My Core List", key="detail_custom_list")
    if c4.button("Add to custom list"):
        engine.add_to_custom_list(list_name.strip() or "My Core List", chosen)
        save_progress(engine.progress)

    if st.button("Text-to-speech placeholder"):
        st.info("TTS integration placeholder: hook this button to a backend speech API later.")


def screen_quiz(engine: VocabAppEngine) -> None:
    st.title("Quiz Mode")
    st.caption("Multiple choice, matching, fill-in-the-blank, and context challenge.")

    available = engine.daily_feed(limit=12)
    if not available:
        st.info("No quiz candidates available.")
        return

    col1, col2 = st.columns([1, 1])
    selected_type = col1.selectbox("Quiz type", ["random", "multiple_choice", "fill_blank", "context_choice", "matching"])

    if st.session_state.quiz_question is None or st.button("New Question", key="new_quiz_q"):
        target = random.choice(available)
        q = build_question(target, available, None if selected_type == "random" else selected_type)
        st.session_state.quiz_question = q
        st.session_state.quiz_word_id = target.id

    q = st.session_state.quiz_question
    target_id = st.session_state.quiz_word_id
    st.markdown(f"### {q['prompt']}")

    if q["type"] in {"multiple_choice", "context_choice"}:
        answer = st.radio("Choose answer", q["options"], key="quiz_choice")
        if st.button("Submit answer", type="primary"):
            correct = answer == q["answer"]
            engine.record_result(target_id, 5 if correct else 2, source="quiz")
            save_progress(engine.progress)
            if correct:
                st.success("Correct. Spaced interval extended.")
            else:
                st.error(f"Not quite. Correct answer: {q['answer']}")

    elif q["type"] == "fill_blank":
        answer = st.text_input("Your answer", key="quiz_fill")
        if st.button("Submit fill", type="primary"):
            correct = answer.strip().lower() == str(q["answer"]).strip().lower()
            engine.record_result(target_id, 5 if correct else 1, source="quiz")
            save_progress(engine.progress)
            if correct:
                st.success("Correct. Nice recall.")
            else:
                st.error(f"Expected: {q['answer']}")

    else:
        st.write("Match each word with one definition.")
        for left in q["left"]:
            choice = st.selectbox(f"Definition for {left}", q["right"], key=f"match_{left}")
            if st.button(f"Check {left}", key=f"check_{left}"):
                correct = choice == q["answer_pairs"][left]
                engine.record_result(target_id, 4 if correct else 2, source="quiz")
                save_progress(engine.progress)
                if correct:
                    st.success("Correct match.")
                else:
                    st.warning("Try again.")

    st.markdown("### Build a sentence mini game")
    sentence = st.text_area("Write one sentence with the current target word.", key="sentence_mini")
    if st.button("Evaluate sentence"):
        word = engine.word_map[target_id].word.lower()
        if word in sentence.lower() and len(sentence.split()) > 6:
            st.success("Strong sentence. You earn +30 bonus XP.")
            engine.progress.xp += 30
            engine.level_up_if_needed()
            save_progress(engine.progress)
        else:
            st.info("Try writing a longer sentence including the target word.")


def screen_review(engine: VocabAppEngine) -> None:
    st.title("Review Mode")
    due = engine.due_reviews()
    st.caption(f"{len(due)} reviews due today.")

    if not due:
        st.success("All clear. No due reviews for now.")
        return

    for word in due[:12]:
        render_word_card(engine, word.id, include_actions=False)
        quality = st.slider(f"How well did you remember '{word.word}'?", 0, 5, 3, key=f"q_{word.id}")
        if st.button("Save review", key=f"save_review_{word.id}"):
            engine.record_result(word.id, quality=quality, source="review")
            save_progress(engine.progress)
            st.success("Review saved.")


def screen_progress(engine: VocabAppEngine) -> None:
    st.title("Progress Analytics")
    snap = engine.progress_snapshot()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words Learned", engine.progress.total_words_learned)
    m2.metric("Mastery Score", f"{sum(s.mastery for s in engine.progress.word_states.values())/len(engine.progress.word_states):.0%}")
    m3.metric("Longest Streak", engine.progress.longest_streak)
    m4.metric("Forgotten Words", len(engine.progress.forgotten_words))

    weak = snap["weak_areas"]
    if weak:
        df = pd.DataFrame(weak)
        fig = px.bar(df, x="category", y="weakness", title="Weak areas (higher means more review needed)", color="weakness", color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Study consistency heatmap")
    heat = pd.DataFrame(snap["heatmap"])
    fig2 = px.imshow(heat, color_continuous_scale=["#f0fdfa", "#14b8a6", "#0f766e"], aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Personal bests")
    st.write(f"Best streak: {engine.progress.longest_streak} days")
    st.write(f"Most words completed in one day: {max([h.get('completed', 0) for h in engine.progress.history] or [0])}")


def screen_saved(engine: VocabAppEngine) -> None:
    st.title("Saved Words and Favorites")
    st.markdown("### Favorites")
    for wid in engine.progress.favorites:
        render_word_card(engine, wid, include_actions=False)

    st.markdown("### Custom lists")
    for list_name, word_ids in engine.progress.custom_lists.items():
        with st.expander(f"{list_name} ({len(word_ids)})"):
            for wid in word_ids:
                st.write(f"- {engine.word_map[wid].word}")

    st.markdown("### Forgotten words")
    for wid in engine.progress.forgotten_words[-12:]:
        st.write(f"- {engine.word_map[wid].word}")


def screen_settings(engine: VocabAppEngine) -> None:
    st.title("Settings")

    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
    goal = st.selectbox("Learning focus", list(GOAL_TO_CATEGORIES.keys()), index=list(GOAL_TO_CATEGORIES.keys()).index(engine.progress.goal))
    daily = st.slider("Daily goal", 5, 20, engine.progress.daily_goal)
    reminder = st.time_input("Daily reminder placeholder", value=None)

    st.markdown("### Search and filters")
    query = st.text_input("Search words instantly")
    difficulty = st.selectbox("Difficulty", ["all", "easy", "medium", "hard"])
    categories = sorted({w.category for w in engine.words})
    category = st.selectbox("Category", ["all"] + categories)
    mastery_bucket = st.selectbox("Mastery", ["all", "new", "learning", "mastered"])

    rows = engine.search(query) if query.strip() else engine.filtered_words(difficulty, category, mastery_bucket)
    st.caption(f"Showing {len(rows)} words")
    for word in rows[:10]:
        st.write(f"- {word.word} ({word.difficulty}, {word.category})")

    if st.button("Save settings", type="primary"):
        engine.progress.goal = goal
        engine.progress.daily_goal = daily
        save_progress(engine.progress)
        st.success("Settings saved.")

    st.caption(f"Reminder placeholder set to: {reminder}")
    st.caption("Offline-ready note: all progress persists to local JSON store for backend-free operation.")


def screen_profile(engine: VocabAppEngine) -> None:
    st.title("User Profile")
    snap = engine.progress_snapshot()

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"## {engine.progress.profile_name}")
        st.write(f"Rank: {snap['rank']}")
        st.write(f"Level: {engine.progress.level}")
        st.write(f"XP: {engine.progress.xp}/{snap['xp_to_next']}")
        st.write(f"Goal: {engine.progress.goal}")

    with c2:
        st.markdown("### Weekly challenge")
        st.info("Finish 30 review cards and 20 quiz questions this week for the Champion badge.")

        st.markdown("### Leaderboard (placeholder)")
        board = pd.DataFrame(
            [
                {"User": engine.progress.profile_name, "XP": engine.progress.xp + engine.progress.level * 100},
                {"User": "Avery", "XP": 1620},
                {"User": "Maya", "XP": 1490},
                {"User": "Liam", "XP": 1310},
            ]
        ).sort_values("XP", ascending=False)
        st.dataframe(board, use_container_width=True)

    st.markdown("### Confusing word pairs mini game")
    pairs = engine.confused_pairs()
    if pairs:
        pair = random.choice(pairs)
        guess = st.radio(
            f"Which word means: {pair['hint']}",
            [pair["word"], pair["confused"]],
            key="pair_guess",
        )
        if st.button("Check pair game"):
            if guess == pair["word"]:
                st.success("Correct distinction. Great precision.")
            else:
                st.warning("Not quite. Review that pair in Word Detail.")


def main() -> None:
    ensure_session()
    inject_theme(st.session_state.dark_mode)
    engine = VocabAppEngine(progress=st.session_state.progress)

    with st.sidebar:
        st.markdown("# LexiLift")
        st.caption("Daily Vocabulary Intelligence")

        page = st.radio(
            "Navigate",
            [
                "Onboarding",
                "Home Dashboard",
                "Daily Words",
                "Word Detail",
                "Quiz Mode",
                "Review Mode",
                "Progress Analytics",
                "Saved Words",
                "Settings",
                "User Profile",
            ],
        )

        st.markdown("---")
        st.metric("Today", f"Day {engine.progress.current_day}")
        st.metric("Streak", f"{engine.progress.streak_days} days")
        st.metric("XP", engine.progress.xp)

    page_handlers = {
        "Onboarding": screen_onboarding,
        "Home Dashboard": screen_home,
        "Daily Words": screen_daily_words,
        "Word Detail": screen_word_detail,
        "Quiz Mode": screen_quiz,
        "Review Mode": screen_review,
        "Progress Analytics": screen_progress,
        "Saved Words": screen_saved,
        "Settings": screen_settings,
        "User Profile": screen_profile,
    }

    page_handlers[page](engine)


if __name__ == "__main__":
    main()
