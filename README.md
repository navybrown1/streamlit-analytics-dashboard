# LexiLift - Vocabulary Learning App

LexiLift is a polished Streamlit app for daily English vocabulary growth with active recall, spaced repetition, contextual usage practice, and gamified progress.

## Product Highlights

- Daily vocabulary feed tuned to learner goal
- Word of the day style flow and review queue
- Rich word detail pages (pronunciation, etymology, synonyms, antonyms, confusion pairs, mnemonic)
- Quiz modes:
  - multiple choice
  - context choice
  - fill in the blank
  - matching
- Spaced repetition logic (SM-2 style adaptation)
- Progress dashboard:
  - words learned
  - mastery score
  - streaks
  - due reviews
  - weak category insights
  - study consistency heatmap
- Gamification:
  - XP and leveling
  - rank labels
  - achievement badges
  - weekly challenge widget
  - leaderboard placeholder
- Personalization:
  - professional, academic, test-prep, or everyday goal
  - filtered and searched word discovery
  - custom lists and favorites
- Smart extras:
  - tap-to-reveal definition recall cards
  - words-you-almost-know panel
  - confusing word pairs mini game
  - build-a-sentence mini game
  - dark mode toggle
  - notification placeholder
  - offline-ready local JSON persistence

## App Screens

- Onboarding
- Home Dashboard
- Daily Words
- Word Detail
- Quiz Mode
- Review Mode
- Progress Analytics
- Saved Words
- Settings
- User Profile

## Component Hierarchy

- app.py
  - Global theme and navigation shell
  - Screen controllers and interactions
- src/vocab/
  - data.py
    - rich mock dataset (20+ words)
  - models.py
    - typed domain models and goal maps
  - srs.py
    - spaced repetition and mastery labels
  - quizzes.py
    - quiz question generators
  - analytics.py
    - rank, badge, heatmap, weak area analytics
  - engine.py
    - orchestration layer for feed, review, scoring, and personalization
  - store.py
    - lightweight persistence to .streamlit/vocab_state.json

## Learning Science Implementation

- Every word has an adaptive review state:
  - interval
  - ease factor
  - repetition count
  - due day
  - mastery
- Wrong answers schedule earlier reviews (interval reset)
- Correct answers increase interval and mastery
- Due reviews are surfaced each day
- Active recall is emphasized via hidden-definition and context-use flows

## Data and Persistence

- Mock vocabulary data is embedded in src/vocab/data.py
- Progress persists to local file:
  - .streamlit/vocab_state.json
- This keeps the app backend-free while enabling future API integration

## Run Locally

```bash
cd /Users/edwinbrown/Documents/streamlit-analytics-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Open the URL printed in terminal (default: http://127.0.0.1:3001).

## Development Notes

- Architecture is modular to support backend integration later.
- You can swap local persistence in src/vocab/store.py with a DB or API client.
- Quiz generation logic is isolated in src/vocab/quizzes.py for future expansion.
