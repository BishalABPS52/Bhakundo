# Bhakundo - Premier League Match Predictor

A comprehensive Premier League match prediction system using machine learning ensemble models.

## 📁 Project Structure

```
bhakundo-predictor/
├── backend/                    # Backend API server
│   ├── api_server.py          # Main FastAPI application
│   ├── auth.py                # Authentication
│   ├── database.py            # Database models
│   ├── ensemble_predictor.py  # Ensemble prediction logic
│   ├── football_api.py        # External API integrations
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment configuration template
│
├── frontend/                   # Next.js frontend application
│   ├── pages/                 # Next.js pages
│   ├── components/            # React components
│   ├── lib/                   # Utility functions & API client
│   ├── styles/                # CSS styles
│   ├── package.json           # Node dependencies
│   └── .env.local.example     # Frontend env template
│
├── required/                   # Shared data, models, and scripts
│   ├── data/                  # All data files
│   │   ├── models/            # ML model files (.pkl)
│   │   │   ├── pl_base_outcome_model.pkl
│   │   │   ├── pl_lineup_model.pkl
│   │   │   └── pl_score_prediction_model.pkl
│   │   ├── raw/               # Raw data files
│   │   │   └── pl/            # Premier League data
│   │   │       ├── pl_2023_historical.csv
│   │   │       ├── pl_2024_historical.csv
│   │   │       ├── pl_2025_26_completed_matches.csv
│   │   │       ├── pl_all_seasons_combined.csv
│   │   │       └── player_data/
│   │   └── processed/         # Processed features
│   │       └── pl/
│   ├── scripts/               # Utility scripts
│   │   ├── pl_fetch_and_update_all.py       # Main update script
│   │   ├── pl_build_features_advanced.py     # Feature engineering
│   │   ├── pl_retrain_all_models_final.py   # Model training
│   │   └── ...
│   ├── dashboard/             # Optional dashboard
│   └── src/                   # Shared source modules
│
└── venv/                       # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Backend Setup

```bash
# Navigate to backend folder
cd backend

# Create and activate virtual environment
python3 -m venv ../venv
source ../venv/bin/activate  # Linux/Mac
# or
..\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Run the server
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Copy and configure environment
cp .env.local.example .env.local
# Edit .env.local if needed

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 📊 Data Management

### Current Data Status

- **2023 Season**: ~380 matches (historical)
- **2024 Season**: ~380 matches (historical)
- **2025-26 Season**: 196 matches completed (as of GW20)
- **Total**: ~956 completed matches

### Fetching Latest Data

To update with latest match results and retrain models:

```bash
# Navigate to required/scripts
cd required/scripts

# Run the comprehensive update script
python pl_fetch_and_update_all.py
```

This script will:
1. ✅ Fetch latest Premier League 2025-26 matches
2. ✅ Update player availability data
3. ✅ Build training features
4. ✅ Retrain all models
5. ✅ Show current status

### Check Current Status

```bash
python pl_fetch_and_update_all.py --status
```

## 🤖 Models

The system uses 3 core models combined in an ensemble:

1. **Base Outcome Model** (`pl_base_outcome_model.pkl`)
   - Predicts match outcome (Home/Draw/Away)
   - Features: team form, standings, H2H history

2. **Lineup Model** (`pl_lineup_model.pkl`)
   - Adjusts predictions based on player availability
   - Features: player quality, lineup strength, formations

3. **Score Prediction Model** (`pl_score_prediction_model.pkl`)
   - Predicts exact scores for both teams
   - Features: offensive/defensive stats, recent form

4. **Ensemble Predictor** (Combined in `ensemble_predictor.py`)
   - Aligns all predictions for consistency
   - Provides confidence scores

## 🌐 URL Fallbacks

### Backend CORS Configuration

The backend accepts requests from (in order):
1. Custom `FRONTEND_URL` (if set)
2. https://bhakundo.vercel.app
3. https://bhakundo-frontend.vercel.app
4. http://localhost:3000

### Frontend API Configuration

The frontend tries these backend URLs (in order):
1. Custom `NEXT_PUBLIC_API_URL` (if set)
2. https://bhakundo-backend1.onrender.com
3. https://bhakundo-backend.onrender.com
4. http://localhost:8000

This ensures the app works in both **production** and **local development**.

## 📋 API Endpoints

### Public Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Get match prediction
- `GET /predictions/history` - View prediction history

### Protected Endpoints (require API key)

- `POST /admin/retrain` - Retrain models
- `GET /admin/stats` - System statistics

## 🔄 Workflow

### After Each Gameweek

1. **Update Data**:
   ```bash
   cd required/scripts
   python pl_fetch_and_update_all.py
   ```

2. **Verify Models**: Models are automatically retrained by the update script

3. **Test Predictions**:
   ```bash
   # Backend should auto-reload if running with --reload
   # Frontend should hot-reload automatically
   ```

### Making Predictions

The ensemble predictor combines all models to provide:
- Match outcome probability (H/D/A)
- Most likely score
- Confidence level
- Key factors (form, lineups, H2H)

## 🛠️ Development

### Adding New Features

1. Update feature engineering in `required/scripts/pl_build_features_advanced.py`
2. Retrain models with `pl_retrain_all_models_final.py`
3. Update API if needed

### Supporting New Leagues

The structure is organized to support multiple leagues:

```
required/data/raw/
├── pl/           # Premier League
└── laliga/       # La Liga (future)
```

Just add league-specific data in new folders.

## 📝 Files Organization

### CSV Files (in `required/data/raw/pl/`)

- `pl_2023_historical.csv` - 2023 season data
- `pl_2024_historical.csv` - 2024 season data
- `pl_2025_26_completed_matches.csv` - Current season matches
- `pl_all_seasons_combined.csv` - All seasons combined for training

### Model Files (in `required/data/models/`)

- `pl_base_outcome_model.pkl` - Base prediction model
- `pl_lineup_model.pkl` - Lineup adjustment model
- `pl_score_prediction_model.pkl` - Score prediction model

## 🚢 Deployment

### Backend (Render/Heroku)

1. Push backend folder to Git
2. Configure environment variables
3. Deploy as Web Service
4. Set start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel/Netlify)

1. Push frontend folder to Git
2. Connect to Vercel/Netlify
3. Configure environment variables
4. Auto-deploy on push

## 📊 Viewing Latest Matches (GW29 Info)

To see the latest completed matches and GW29 fixtures:

```bash
cd required/data/raw/pl
head -1 pl_all_seasons_combined.csv  # Headers
grep "2025-26,29" pl_all_seasons_combined.csv  # GW29 matches
grep "2025-26.*FINISHED" pl_all_seasons_combined.csv | tail -20  # Latest completed
```

## ⚡ Performance

- **Prediction Time**: < 100ms
- **Model Size**: ~3.8 MB total
- **API Response**: < 200ms average

## 🔐 Security

- API key authentication for admin endpoints
- CORS properly configured
- Environment variables for sensitive data
- SQLite for local, PostgreSQL for production

## 📞 Support

For issues or questions:
- Check logs in `required/data/logs/`
- Review model training output
- Test API endpoints with `/health`

## 📜 License

Proprietary - Bishal Shrestha

---

**Last Updated**: March 8, 2026  
**Current Season**: 2025-26  
**Latest Gameweek**: GW20 (196 matches completed)
