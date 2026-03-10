"""Streamlit Dashboard for Premier League Predictions"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.prediction_system import PredictionSystem
from src.data.database import get_session, Team

# Page config
st.set_page_config(
    page_title="Premier League Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_prediction_system():
    """Load prediction system (cached)"""
    return PredictionSystem()


@st.cache_data(ttl=3600)
def load_teams():
    """Load teams from database"""
    session = get_session()
    teams = session.query(Team).all()
    session.close()
    return {team.team_id: team.team_name for team in teams}


def display_prediction(prediction):
    """Display prediction results"""
    if not prediction:
        st.error("No prediction available")
        return
    
    match_info = prediction['match_info']
    outcome = prediction['outcome_prediction']
    score = prediction['score_prediction']
    additional = prediction['additional_predictions']
    factors = prediction['key_factors']
    
    # Match header
    st.markdown(f"""
        <h2 style='text-align: center;'>
        {match_info['home_team']} 🆚 {match_info['away_team']}
        </h2>
        """, unsafe_allow_html=True)
    
    # Outcome probabilities
    st.subheader("⚡ Match Outcome Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Home Win",
            value=f"{outcome['home_win_probability']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Draw",
            value=f"{outcome['draw_probability']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Away Win",
            value=f"{outcome['away_win_probability']*100:.1f}%"
        )
    
    # Probability chart
    probs_df = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [
            outcome['home_win_probability'],
            outcome['draw_probability'],
            outcome['away_win_probability']
        ]
    })
    
    fig = px.bar(probs_df, x='Outcome', y='Probability', 
                 title='Outcome Probabilities',
                 color='Probability',
                 color_continuous_scale='blues')
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predicted outcome
    st.info(f"**Predicted Outcome:** {outcome['predicted_outcome']} | **Confidence:** {outcome['confidence']}")
    
    # Score prediction
    st.subheader("⚽ Score Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
            <h3 style='text-align: center;'>{score['most_likely_score']}</h3>
            <p style='text-align: center;'>Most Likely Score</p>
            <p style='text-align: center;'>Probability: {score['probability']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Top 5 Most Likely Scorelines:**")
        scores_df = pd.DataFrame(score['top_5_scorelines'])
        scores_df['probability_pct'] = scores_df['probability_pct'].round(2)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)
    
    # Additional predictions
    st.subheader("📊 Additional Markets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Over/Under 2.5 Goals**")
        over_under_df = pd.DataFrame({
            'Market': ['Over 2.5', 'Under 2.5'],
            'Probability': [additional['over_2_5'], additional['under_2_5']]
        })
        fig_ou = px.pie(over_under_df, names='Market', values='Probability',
                        title='Over/Under 2.5 Goals')
        st.plotly_chart(fig_ou, use_container_width=True)
    
    with col2:
        st.write("**Both Teams To Score**")
        btts_df = pd.DataFrame({
            'Market': ['BTTS Yes', 'BTTS No'],
            'Probability': [additional['btts_yes'], additional['btts_no']]
        })
        fig_btts = px.pie(btts_df, names='Market', values='Probability',
                          title='Both Teams To Score')
        st.plotly_chart(fig_btts, use_container_width=True)
    
    # Key factors
    st.subheader("🔑 Key Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{match_info['home_team']}**")
        st.write(f"- Recent Form: `{factors['home_recent_form']}`")
        st.write(f"- Avg xG: {factors['home_xg_avg']:.2f}")
    
    with col2:
        st.write(f"**{match_info['away_team']}**")
        st.write(f"- Recent Form: `{factors['away_recent_form']}`")
        st.write(f"- Avg xG: {factors['away_xg_avg']:.2f}")


def main():
    """Main dashboard function"""
    
    # Title
    st.title("⚽ Premier League Match Prediction System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        page = st.radio(
            "Select Page",
            ["Single Match Prediction", "Upcoming Fixtures", "About"]
        )
    
    try:
        # Load system
        system = load_prediction_system()
        teams = load_teams()
        
        if page == "Single Match Prediction":
            st.header("Single Match Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                home_team_id = st.selectbox(
                    "Home Team",
                    options=list(teams.keys()),
                    format_func=lambda x: teams[x]
                )
            
            with col2:
                away_team_id = st.selectbox(
                    "Away Team",
                    options=list(teams.keys()),
                    format_func=lambda x: teams[x]
                )
            
            if st.button("Generate Prediction", type="primary"):
                if home_team_id == away_team_id:
                    st.error("Please select different teams")
                else:
                    with st.spinner("Generating prediction..."):
                        try:
                            prediction = system.predict_match(home_team_id, away_team_id)
                            display_prediction(prediction)
                        except Exception as e:
                            st.error(f"Error generating prediction: {e}")
        
        elif page == "Upcoming Fixtures":
            st.header("Upcoming Fixtures Predictions")
            
            limit = st.slider("Number of fixtures", 1, 20, 10)
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    try:
                        predictions = system.predict_upcoming_fixtures(limit=limit)
                        
                        if not predictions:
                            st.warning("No upcoming fixtures found in database")
                        else:
                            for i, pred in enumerate(predictions, 1):
                                with st.expander(
                                    f"GW {pred.get('gameweek', '?')}: "
                                    f"{pred['match_info']['home_team']} vs "
                                    f"{pred['match_info']['away_team']}"
                                ):
                                    display_prediction(pred)
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
        
        else:  # About page
            st.header("About This System")
            
            st.markdown("""
            ### 🎯 Premier League Prediction System
            
            This system uses machine learning to predict:
            - **Match Outcomes**: Home Win / Draw / Away Win with probabilities
            - **Exact Scorelines**: Most likely scores using xG data
            - **Additional Markets**: Over/Under, Both Teams To Score, etc.
            
            ### 📊 Features
            
            The model uses over 50+ features including:
            - Team form (last 5, 10, 15 games)
            - Expected Goals (xG) statistics
            - Home/Away performance splits
            - Head-to-head records
            - Form trajectories and trends
            
            ### 🤖 Models
            
            - **Outcome Model**: Ensemble of XGBoost, LightGBM, and CatBoost
            - **Score Model**: Poisson distribution based on xG
            
            ### 📈 Performance
            
            Target metrics:
            - Match Outcome Accuracy: >55%
            - Log Loss: <1.0
            - Exact Score Prediction: >18%
            
            ### ⚠️ Disclaimer
            
            This system is for **educational and entertainment purposes only**.
            Please gamble responsibly.
            """)
    
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Please ensure:")
        st.info("1. Database is set up: `python scripts/setup_database.py`")
        st.info("2. Data is collected: `python scripts/download_historical_data.py`")
        st.info("3. Models are trained: `python scripts/train_models.py`")


if __name__ == "__main__":
    main()
