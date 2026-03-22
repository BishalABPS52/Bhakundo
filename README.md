
<p align="center">
  <img src="images/bhakundo.png" alt="Bhakundo Logo" width="200"/>
</p>

<p align="center">
  <strong>Bhakundo – Prepare, Predict & Play</strong><br>
</p>

<p align="center">
  Live at: <a href="https://bhakundo.vercel.app">https://bhakundo.vercel.app</a>
</p>

---

## About

Bhakundo is a Football match prediction platform powered by a 3-model machine learning ensemble. It gives football fans a single place to view live fixtures, standings, and get predictions for upcoming matches with predicted scorelines, win probabilities.

---

## Screenshots

### Landing Page
![Landing Page](images/landingpage.png)

### Predictor
![Predictor](images/predictor.png)

### Fixtures & Results
![Fixtures](images/fixtures.png)

### Standings
![Standings](images/standings.png)

### Help
![Help](images/help.png)

---

## Key Features

### For Football Enthsiasts
- Get match predictions with win/draw/loss probabilities
- View expected scorelines aligned to outcome predictions
- Browse live Football fixtures and results by gameweek
- View the current Premier League standings table

### Prediction Engine
- **Base Model** — CatBoost classifier trained on 133 features 
- **Score Model** — LightGBM + XGBoost blend predicting home and away goals via Poisson regression
- **Lineup Model** — CatBoost classifier with formation perturbation augmentation
- **Ensemble Logic** — BHAKUNDO priority system: Base + Lineup agreement triggers highest-confidence predictions; full disagreement falls back to formation-weighted priority

---

## Tech Stack

| Layer | Technology | Hosted On |
|---|---|---|
| Frontend | <img src="https://img.shields.io/badge/React-%2361DAFB.svg?style=for-the-badge&logo=react&logoColor=black" height="30"> <img src="https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white" height="30"> <img src="https://img.shields.io/badge/TailwindCSS-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white" height="30"> | <img src="https://img.shields.io/badge/Vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white" height="30"> |
| Backend | <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" height="30"> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" height="30"> | <img src="https://img.shields.io/badge/Render-%2346E3B7.svg?style=for-the-badge&logo=render&logoColor=white" height="30"> |
| Database | <img src="https://img.shields.io/badge/PostgreSQL-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white" height="30"> | <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=black" height="30"> |
| ML Models | <img src="https://img.shields.io/badge/CatBoost-FFD700?style=for-the-badge&logo=yandex&logoColor=black" height="30"> <img src="https://img.shields.io/badge/XGBoost-EC6C00?style=for-the-badge&logo=xgboost&logoColor=white" height="30"> <img src="https://img.shields.io/badge/LightGBM-00B050?style=for-the-badge&logo=microsoft&logoColor=white" height="30"> | — |
| Live Data | <img src="https://img.shields.io/badge/FPL%20API-38003C?style=for-the-badge&logo=premierleague&logoColor=white" height="30"> <img src="https://img.shields.io/badge/Football--Data.org-004C97?style=for-the-badge&logo=football&logoColor=white" height="30"> | — |

---

## Project Structure

```
bishalabps52-bhakundo/
├── backend/
│   ├── api_server.py            # Main FastAPI app — all endpoints
│   ├── ensemble_predictor.py    # BHAKUNDO priority ensemble logic
│   ├── model_classes.py         
│   ├── poisson_score_predictor.py  # Poisson score probability calculator
│   ├── football_api.py          # FPL + Football-Data.org API integration
│   ├── database.py              # SQLAlchemy models (Prediction, Actual)
│   ├── auth.py                  # API key + admin auth middleware
│   ├── requirements.txt
│   ├── render.yaml
│   └── scripts/
│       ├── pl_retrain_all_models_final.py  # Full training pipeline
│       ├── fetch_newgw.py                  # Fetch new GW results + retrain
│       ├── comprehensive_feature_engineering.py  # 125+ features builder
│       └── train_score_model_ensemble.py   # Score model training
├── frontend/
│   └── src/
│       ├── pages/               # index, predictor, fixtures, standings
│       ├── components/          # Navbar, Footer, FixturesPL, StandingsPL
│       ├── lib/                 # API config and fetcher
│       └── styles/
└── required/
    └── data/
        └── models/              # Trained .pkl model files
```

---

## Prediction Engine — How It Works

```mermaid
flowchart TD
    A[Upcoming Match] --> B[Feature Engineering: 125+ Features]
    B --> C[Base Model: CatBoost + Platt Scaling]
    B --> D[Score Model: LightGBM + XGBoost Blend]
    B --> E[Lineup Model: CatBoost + Formation Augment]

    C --> F{BHAKUNDO\nEnsemble Logic}
    D --> F
    E --> F

    F --> G{Agreement Check}

    G -->|All 3 Agree| H[Highest Confidence: All Models Agree]
    G -->|Base + Lineup Agree| I[ BHAKUNDO PREDICTS: High Confidence]
    G -->|2 of 3 Agree| J[Majority Consensus :Medium Confidence]
    G -->|All Disagree| K{Custom Formation?}

    K -->|Yes| L[Lineup Priority: Edited Formation]
    K -->|No| M[Base Priority; Default Formation]

    H --> N[Final Prediction]
    I --> N
    J --> N
    L --> N
    M --> N

    N --> O[Score Alignment; Score Must Match Outcome]
    O --> P[Output: Outcome + Score+ Probabilities + Confidence]

    style A fill:#38003C,color:#fff,stroke:none
    style B fill:#1E40AF,color:#fff,stroke:none
    style C fill:#065F46,color:#fff,stroke:none
    style D fill:#065F46,color:#fff,stroke:none
    style E fill:#065F46,color:#fff,stroke:none
    style F fill:#7C3AED,color:#fff,stroke:none
    style G fill:#1F2937,color:#fff,stroke:none
    style H fill:#047857,color:#fff,stroke:none
    style I fill:#B45309,color:#fff,stroke:none
    style J fill:#0E7490,color:#fff,stroke:none
    style K fill:#1F2937,color:#fff,stroke:none
    style L fill:#9D174D,color:#fff,stroke:none
    style M fill:#1E3A8A,color:#fff,stroke:none
    style N fill:#4F46E5,color:#fff,stroke:none
    style O fill:#374151,color:#fff,stroke:none
    style P fill:#38003C,color:#fff,stroke:none
```

---

## Model Details

**Training data:** 1000+ finished Premier League matches (2023–2026)  
**Feature highlights:**  Venue stats, H2H, xG, Player Form, Home Form , Away Form , Goal For , Goal Against and many more

---

## Live Data Sources

- **Fantasy Premier League API** — fixtures, gameweek, team names 
- **Football-Data.org API** — match results, standings, live scores
- **Fallback** — formatted 2025-26 season CSV loaded from disk if APIs are unavailable

---

## Coming Soon

- User accounts with prediction history and accuracy tracking
- Prediction leaderboard across gameweeks
- La Liga, Bundesliga, UEFA Champions League and other football leagues support
- Automated retraining after each gameweek 

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- PostgreSQL

### Installation

#### Clone the repo
```bash
git clone https://github.com/BishalABPS52/bhakundo.git
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs on http://localhost:3000

#### Backend
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.api_server:app --reload --port 8000
```
Runs on http://localhost:8000

## Developer

### Built by [Bishal Shrestha](https://bishalshrestha52.com.np)

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BishalABPS52)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bishal-shrestha-2b05b1302/)
