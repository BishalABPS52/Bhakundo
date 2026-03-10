"""
Premier League Match Prediction Dashboard
Compares predictions from Base Model and Lineup-Based Model
Includes score predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lineup_models import LineupPredictor
from scripts.train_final_model import ManualEnsemble

# Page config
st.set_page_config(
    page_title="Premier League Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #37003c;
        padding: 20px;
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #37003c;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .formation-box {
        background: #37003c;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #37003c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both prediction models"""
    base_model = joblib.load('data/models/outcome_model_final.pkl')
    lineup_model = joblib.load('data/models/lineup_result_model.pkl')
    return base_model, lineup_model


@st.cache_data
def load_match_data():
    """Load recent match data"""
    df = pd.read_csv('data/processed/training_dataset_advanced.csv')
    return df


@st.cache_data
def load_raw_matches():
    """Load raw match data with lineups and injuries"""
    return pd.read_csv('data/raw/matches_combined.csv')


def get_team_features(team_name, df):
    """Extract recent form features for a team"""
    recent = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].tail(5)
    
    points = []
    goals_for = []
    goals_against = []
    
    for _, match in recent.iterrows():
        if match['home_team'] == team_name:
            gf, ga = match['home_goals'], match['away_goals']
        else:
            gf, ga = match['away_goals'], match['home_goals']
        
        goals_for.append(gf)
        goals_against.append(ga)
        
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)
    
    return {
        'ppg': np.mean(points) if points else 1.5,
        'goals_for': sum(goals_for),
        'goals_against': sum(goals_against),
        'wins': sum(1 for p in points if p == 3),
        'draws': sum(1 for p in points if p == 1),
        'losses': sum(1 for p in points if p == 0)
    }


def get_predicted_lineup(team_name, formation):
    """Get predicted starting lineup based on formation"""
    # Mock lineup data - in production, this would come from squad data
    formations_data = {
        '4-3-3': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Back', 'Center Back', 'Center Back', 'Left Back'],
            'MID': ['Central Midfielder', 'Central Midfielder', 'Central Midfielder'],
            'ATT': ['Right Winger', 'Striker', 'Left Winger']
        },
        '4-2-3-1': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Back', 'Center Back', 'Center Back', 'Left Back'],
            'MID': ['Defensive Midfielder', 'Defensive Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Left Midfielder'],
            'ATT': ['Striker']
        },
        '3-4-3': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Center Back', 'Center Back', 'Left Center Back'],
            'MID': ['Right Wing Back', 'Central Midfielder', 'Central Midfielder', 'Left Wing Back'],
            'ATT': ['Right Winger', 'Striker', 'Left Winger']
        },
        '4-4-2': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Back', 'Center Back', 'Center Back', 'Left Back'],
            'MID': ['Right Midfielder', 'Central Midfielder', 'Central Midfielder', 'Left Midfielder'],
            'ATT': ['Striker', 'Striker']
        },
        '3-5-2': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Center Back', 'Center Back', 'Left Center Back'],
            'MID': ['Right Wing Back', 'Central Midfielder', 'Central Midfielder', 'Central Midfielder', 'Left Wing Back'],
            'ATT': ['Striker', 'Striker']
        },
        '5-3-2': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Wing Back', 'Right Center Back', 'Center Back', 'Left Center Back', 'Left Wing Back'],
            'MID': ['Central Midfielder', 'Central Midfielder', 'Central Midfielder'],
            'ATT': ['Striker', 'Striker']
        },
        '4-5-1': {
            'GK': ['Goalkeeper'],
            'DEF': ['Right Back', 'Center Back', 'Center Back', 'Left Back'],
            'MID': ['Right Midfielder', 'Central Midfielder', 'Central Midfielder', 'Central Midfielder', 'Left Midfielder'],
            'ATT': ['Striker']
        }
    }
    
    return formations_data.get(formation, formations_data['4-3-3'])


def get_team_injuries(team_name):
    """Get injury information for a team - mock data"""
    # In production, this would fetch from injury/suspension API
    mock_injuries = {
        'Manchester City FC': [
            {'player': 'Kevin De Bruyne', 'status': 'Doubtful', 'injury': 'Hamstring'},
        ],
        'Liverpool FC': [
            {'player': 'Trent Alexander-Arnold', 'status': 'Out', 'injury': 'Knee'},
        ],
        'Arsenal FC': [
            {'player': 'Gabriel Jesus', 'status': 'Doubtful', 'injury': 'Knee'},
        ]
    }
    
    return mock_injuries.get(team_name, [])


def predict_score(probabilities, home_features, away_features):
    """Predict match score based on outcome probabilities and team features"""
    # Expected goals based on recent form
    home_xg = home_features['goals_for'] / 5  # Average per match
    away_xg = away_features['goals_for'] / 5
    
    # Adjust based on prediction
    if probabilities[0] > 0.5:  # Home win likely
        home_goals = int(round(home_xg * 1.2))
        away_goals = int(round(away_xg * 0.8))
    elif probabilities[2] > 0.5:  # Away win likely
        home_goals = int(round(home_xg * 0.8))
        away_goals = int(round(away_xg * 1.2))
    else:  # Draw likely
        home_goals = int(round(home_xg))
        away_goals = home_goals
    
    # Ensure sensible ranges
    home_goals = max(0, min(home_goals, 4))
    away_goals = max(0, min(away_goals, 4))
    
    # Adjust for draw prediction
    if probabilities[1] > 0.3 and home_goals != away_goals:
        avg = (home_goals + away_goals) // 2
        home_goals = away_goals = avg
    
    return home_goals, away_goals


def create_probability_chart(probabilities, title):
    """Create a bar chart for outcome probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Home Win', 'Draw', 'Away Win'],
            y=[probabilities[0], probabilities[1], probabilities[2]],
            marker_color=['#00ff87', '#ffd700', '#ff6b6b'],
            text=[f"{p:.1%}" for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">⚽ Premier League Match Predictor</div>', 
                unsafe_allow_html=True)
    
    # Load models and data
    try:
        base_model, lineup_model = load_models()
        df = load_match_data()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Match Configuration")
    
    # Get unique teams
    teams = sorted(df['home_team'].unique())
    
    # Team selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=teams.index("Liverpool FC") if "Liverpool FC" in teams else 0)
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=teams.index("Manchester City FC") if "Manchester City FC" in teams else 1)
    
    # Formation selection
    st.sidebar.markdown("### 📋 Formations (Optional)")
    formations = ['Auto-Predict', '4-3-3', '4-2-3-1', '3-4-3', '4-4-2', '3-5-2', '5-3-2', '4-5-1']
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        home_formation = st.selectbox("Home Formation", formations)
    with col2:
        away_formation = st.selectbox("Away Formation", formations)
    
    # Predict button
    if st.sidebar.button("🎯 Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select different teams!")
            return
        
        # Get team features
        home_features = get_team_features(home_team, df)
        away_features = get_team_features(away_team, df)
        
        # Main content
        st.markdown(f"## {home_team} vs {away_team}")
        
        # Recent Form
        st.markdown("### 📊 Recent Form (Last 5 Matches)")
        col1, col2, col3 = st.columns([1, 0.5, 1])
        
        with col1:
            st.markdown(f"**{home_team}**")
            st.metric("Points Per Game", f"{home_features['ppg']:.2f}")
            st.metric("Goals Scored", home_features['goals_for'])
            st.metric("Goals Conceded", home_features['goals_against'])
            st.write(f"W: {home_features['wins']} | D: {home_features['draws']} | L: {home_features['losses']}")
        
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 50px; font-size: 2rem;'>⚔️</div>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**{away_team}**")
            st.metric("Points Per Game", f"{away_features['ppg']:.2f}")
            st.metric("Goals Scored", away_features['goals_for'])
            st.metric("Goals Conceded", away_features['goals_against'])
            st.write(f"W: {away_features['wins']} | D: {away_features['draws']} | L: {away_features['losses']}")
        
        st.markdown("---")
        
        # Predictions from both models
        st.markdown("## 🔮 Match Predictions")
        
        # Prepare features (simplified - using most recent similar match)
        latest = df[(df['home_team'] == home_team) | (df['away_team'] == away_team)].tail(1)
        
        if not latest.empty:
            # Base Model Prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Base Model")
                st.caption("86 features: Form, momentum, venue stats")
                
                # Create feature vector for base model
                X_base = latest[base_model['feature_names']].iloc[0:1]
                X_base_scaled = base_model['scaler'].transform(X_base)
                
                # Predict
                base_probs = base_model['ensemble'].predict_proba(X_base_scaled)[0]
                base_pred = base_model['ensemble'].predict(X_base_scaled)[0]
                
                # Predict score
                home_score, away_score = predict_score(base_probs, home_features, away_features)
                
                # Display score with team names
                st.markdown(f"#### 📊 Predicted Score")
                col_a, col_b, col_c = st.columns([2, 1, 2])
                with col_a:
                    st.markdown(f"<div style='text-align: right; font-size: 1.2rem; font-weight: bold;'>{home_team}</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="score-display" style="font-size: 2.5rem;">
                        {home_score} - {away_score}
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"<div style='text-align: left; font-size: 1.2rem; font-weight: bold;'>{away_team}</div>", unsafe_allow_html=True)
                
                st.markdown("")
                outcomes = ['HOME WIN', 'DRAW', 'AWAY WIN']
                st.markdown(f"**Prediction: {outcomes[base_pred]}**")
                st.markdown(f"**Confidence: {base_probs[base_pred]:.1%}**")
                
                # Probability chart
                fig = create_probability_chart(base_probs, "Outcome Probabilities")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Lineup-Based Model")
                st.caption("97 features: Base + formations & tactics")
                
                # Predict formations
                predictor = LineupPredictor()
                
                if home_formation == 'Auto-Predict':
                    home_form = predictor.predict_formation({
                        'ppg_last5': home_features['ppg'],
                        'goals_for_last5': home_features['goals_for'],
                        'goals_against_last5': home_features['goals_against']
                    })
                else:
                    home_form = home_formation
                
                if away_formation == 'Auto-Predict':
                    away_form = predictor.predict_formation({
                        'ppg_last5': away_features['ppg'],
                        'goals_for_last5': away_features['goals_for'],
                        'goals_against_last5': away_features['goals_against']
                    })
                else:
                    away_form = away_formation
                
                # Display formations
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="formation-box">{home_form}</div>', 
                               unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<div class="formation-box">{away_form}</div>', 
                               unsafe_allow_html=True)
                
                # Encode formations
                home_encoded = predictor.encode_formation(home_form)
                away_encoded = predictor.encode_formation(away_form)
                
                # Create feature vector for lineup model
                # Get base features from the latest match
                base_feature_names = [f for f in lineup_model['feature_names'] 
                                     if f not in ['home_def_line', 'home_mid_line', 'home_att_line',
                                                  'away_def_line', 'away_mid_line', 'away_att_line',
                                                  'def_vs_att_home', 'def_vs_att_away', 'mid_battle',
                                                  'home_aggression', 'away_aggression']]
                
                X_lineup = latest[base_feature_names].copy()
                
                # Add lineup features
                X_lineup['home_def_line'] = home_encoded['def_line']
                X_lineup['home_mid_line'] = home_encoded['mid_line']
                X_lineup['home_att_line'] = home_encoded['att_line']
                X_lineup['away_def_line'] = away_encoded['def_line']
                X_lineup['away_mid_line'] = away_encoded['mid_line']
                X_lineup['away_att_line'] = away_encoded['att_line']
                
                X_lineup['def_vs_att_home'] = home_encoded['def_line'] - away_encoded['att_line']
                X_lineup['def_vs_att_away'] = away_encoded['def_line'] - home_encoded['att_line']
                X_lineup['mid_battle'] = home_encoded['mid_line'] - away_encoded['mid_line']
                X_lineup['home_aggression'] = home_encoded['att_line'] + home_encoded['mid_line'] - home_encoded['def_line']
                X_lineup['away_aggression'] = away_encoded['att_line'] + away_encoded['mid_line'] - away_encoded['def_line']
                
                # Reorder columns to match training
                X_lineup = X_lineup[lineup_model['feature_names']]
                
                X_lineup_scaled = lineup_model['scaler'].transform(X_lineup.iloc[0:1])
                
                # Predict
                lineup_probs = lineup_model['lineup_model'].predict_proba(X_lineup_scaled)[0]
                lineup_pred = lineup_model['lineup_model'].predict(X_lineup_scaled)[0]
                
                # Predict score (adjusted for formations)
                home_score_l, away_score_l = predict_score(lineup_probs, home_features, away_features)
                
                # Adjust for aggression
                if X_lineup['home_aggression'].iloc[0] > X_lineup['away_aggression'].iloc[0]:
                    home_score_l = min(home_score_l + 1, 5)
                elif X_lineup['away_aggression'].iloc[0] > X_lineup['home_aggression'].iloc[0]:
                    away_score_l = min(away_score_l + 1, 5)
                
                # Display score with team names
                st.markdown(f"#### 📊 Predicted Score")
                col_a, col_b, col_c = st.columns([2, 1, 2])
                with col_a:
                    st.markdown(f"<div style='text-align: right; font-size: 1.2rem; font-weight: bold;'>{home_team}</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="score-display" style="font-size: 2.5rem;">
                        {home_score_l} - {away_score_l}
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"<div style='text-align: left; font-size: 1.2rem; font-weight: bold;'>{away_team}</div>", unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown(f"**Prediction: {outcomes[lineup_pred]}**")
                st.markdown(f"**Confidence: {lineup_probs[lineup_pred]:.1%}**")
                
                # Probability chart
                fig = create_probability_chart(lineup_probs, "Outcome Probabilities")
                st.plotly_chart(fig, use_container_width=True)
        
        # Predicted Lineups Section
        st.markdown("---")
        st.markdown("### 📋 Predicted Starting Lineups")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {home_team}")
            st.markdown(f"**Formation: {home_form}**")
            
            home_lineup = get_predicted_lineup(home_team, home_form)
            
            # Display lineup by position
            for position, players in home_lineup.items():
                st.markdown(f"**{position}:**")
                for player in players:
                    st.write(f"• {player}")
            
            # Show injuries
            home_injuries = get_team_injuries(home_team)
            if home_injuries:
                st.markdown("**⚠️ Injuries/Doubts:**")
                for injury in home_injuries:
                    status_color = "🔴" if injury['status'] == 'Out' else "🟡"
                    st.write(f"{status_color} {injury['player']} - {injury['injury']} ({injury['status']})")
        
        with col2:
            st.markdown(f"#### {away_team}")
            st.markdown(f"**Formation: {away_form}**")
            
            away_lineup = get_predicted_lineup(away_team, away_form)
            
            # Display lineup by position
            for position, players in away_lineup.items():
                st.markdown(f"**{position}:**")
                for player in players:
                    st.write(f"• {player}")
            
            # Show injuries
            away_injuries = get_team_injuries(away_team)
            if away_injuries:
                st.markdown("**⚠️ Injuries/Doubts:**")
                for injury in away_injuries:
                    status_color = "🔴" if injury['status'] == 'Out' else "🟡"
                    st.write(f"{status_color} {injury['player']} - {injury['injury']} ({injury['status']})")
        
        # Tactical Analysis (Lineup Model)
        st.markdown("---")
        st.markdown("### ⚔️ Tactical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mid_battle = X_lineup['mid_battle'].iloc[0]
            st.metric("Midfield Battle", 
                     f"{'+' if mid_battle > 0 else ''}{mid_battle:.0f}",
                     f"Advantage: {home_team if mid_battle > 0 else away_team if mid_battle < 0 else 'Even'}")
        
        with col2:
            home_agg = X_lineup['home_aggression'].iloc[0]
            away_agg = X_lineup['away_aggression'].iloc[0]
            st.metric("Formation Style", 
                     "Attacking" if abs(home_agg + away_agg) > 4 else "Balanced",
                     f"Home: {home_agg:+.0f} | Away: {away_agg:+.0f}")
        
        with col3:
            def_match = X_lineup['def_vs_att_home'].iloc[0]
            st.metric("Defensive Setup",
                     "Strong" if def_match > 0 else "Weak" if def_match < 0 else "Balanced",
                     f"Home: {def_match:+.0f}")
        
        # Model Comparison
        st.markdown("---")
        st.markdown("### 📈 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['Base Model', 'Lineup Model'],
            'Home Win': [base_probs[0], lineup_probs[0]],
            'Draw': [base_probs[1], lineup_probs[1]],
            'Away Win': [base_probs[2], lineup_probs[2]],
            'Predicted Score': [f"{home_score}-{away_score}", f"{home_score_l}-{away_score_l}"]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Confidence comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Base Model', x=['Home Win', 'Draw', 'Away Win'], 
                             y=base_probs, marker_color='#00ff87'))
        fig.add_trace(go.Bar(name='Lineup Model', x=['Home Win', 'Draw', 'Away Win'], 
                             y=lineup_probs, marker_color='#60efff'))
        
        fig.update_layout(
            title='Probability Comparison',
            yaxis_title='Probability',
            yaxis=dict(tickformat='.0%'),
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Premier League Match Predictor! ⚽
        
        This dashboard uses **two advanced machine learning models** to predict match outcomes:
        
        ### 🎯 Base Model (86 features)
        - Recent form and points per game
        - Goals scored and conceded
        - Home/away performance
        - Head-to-head records
        - Momentum and trends
        - Rest days between matches
        
        ### 🎯 Lineup-Based Model (97 features)
        - All features from Base Model
        - **Formation predictions** (4-3-3, 4-2-3-1, etc.)
        - **Tactical matchups** (defense vs attack)
        - **Midfield battle** (numerical advantage)
        - **Formation aggression** (attacking intent)
        
        ### 📊 What You Get:
        - ✅ **Predicted scoreline** for both models
        - ✅ **Win/Draw/Loss probabilities**
        - ✅ **Tactical analysis** with formations
        - ✅ **Recent form comparison**
        - ✅ **Side-by-side model comparison**
        
        **👈 Select teams from the sidebar to get started!**
        """)
        
        # Show model stats
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Base Model Stats")
            st.metric("Test Accuracy", f"{base_model['test_accuracy']:.1%}")
            st.metric("Training Date", base_model['training_date'])
            st.metric("Features", base_model['n_features'])
        
        with col2:
            st.markdown("### 📊 Lineup Model Stats")
            st.metric("Test Accuracy", f"{lineup_model['test_accuracy']:.1%}")
            st.metric("Training Date", lineup_model['training_date'])
            st.metric("Features", f"{lineup_model['n_features']} (+11 lineup)")


if __name__ == "__main__":
    main()
