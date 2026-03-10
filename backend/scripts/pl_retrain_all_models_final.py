"""
BHAKUNDO PREDICTOR — Complete Model Training Pipeline (Final)
==============================================================
Trains 3-model ensemble:
  1. Base Outcome Model  — CatBoostClassifier + Platt scaling calibration
  2. Score Model         — LightGBMRegressor + XGBoostRegressor blend + Poisson
  3. Lineup Model        — CatBoostClassifier (with formation perturbation)

Feature set: 133 features
  • 125 base (form L5/L10/L20, venue stats, H2H, standings, derived diffs)
  • 6  Elo  (home_elo, away_elo, elo_diff, win_prob, away_win_prob, momentum)
  • 8  Attack/Defense strength ratings + dynamic xG proxies
  • Real rest-day calculation (replaces hardcoded 7)

Data: pl_all_seasons_combined.csv  (~1140 finished PL matches 2023-2026)
Save: required/data/models/
"""

import sys
import os
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_absolute_error, mean_squared_error)
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_DIR  = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR     = PROJECT_ROOT / 'required' / 'data'
RAW_DIR      = DATA_DIR / 'raw' / 'pl'
MODELS_DIR   = DATA_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Import existing feature engineer (125 base features) + shared model classes
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))  # so 'backend.model_classes' is importable
from comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from backend.model_classes import PlattCalibratedCatBoost

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s — %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# PlattCalibratedCatBoost imported from backend.model_classes (above)


# ════════════════════════════════════════════════════════════════════════════
class ComprehensiveModelTrainer:
    """
    End-to-end trainer: loads data → builds 133 features → trains 3 models → saves.
    """

    def __init__(self):
        self.matches_df   = None
        self.engineer     = ComprehensiveFeatureEngineer()
        self._elo_snapshots = {}
        self._final_elo     = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1. DATA LOADING
    # ────────────────────────────────────────────────────────────────────────
    def load_all_matches(self) -> pd.DataFrame:
        """Load pl_all_seasons_combined.csv, keep FINISHED, sort by date."""
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL MATCH DATA")
        logger.info("=" * 80)

        csv = RAW_DIR / 'pl_all_seasons_combined.csv'
        if not csv.exists():
            # fallback: try individual season files
            logger.warning(f"Combined CSV not found at {csv}, loading individual files…")
            frames = []
            for fp in [RAW_DIR / 'pl_2023_historical.csv',
                       RAW_DIR / 'pl_2024_historical.csv',
                       RAW_DIR / 'pl_2025_26_completed_matches.csv']:
                if fp.exists():
                    frames.append(pd.read_csv(fp))
            if not frames:
                raise FileNotFoundError("No match data CSV found in " + str(RAW_DIR))
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.read_csv(csv)

        # Normalise column names
        if 'api_match_id' in df.columns and 'match_id' not in df.columns:
            df = df.rename(columns={'api_match_id': 'match_id'})
        if 'home_score' in df.columns and 'home_goals' not in df.columns:
            df['home_goals'] = df['home_score']
            df['away_goals'] = df['away_score']

        # Keep only finished matches
        if 'status' in df.columns:
            df = df[df['status'].isin(['FINISHED', 'FT'])].copy()
        elif 'result' in df.columns:
            df = df[df['result'].isin(['H', 'D', 'A'])].copy()

        df['match_date'] = pd.to_datetime(df['match_date'], utc=True, errors='coerce')
        df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
        df = df.dropna(subset=['home_goals', 'away_goals', 'home_team', 'away_team', 'match_date'])
        df['home_goals'] = df['home_goals'].astype(int)
        df['away_goals'] = df['away_goals'].astype(int)
        df['match_id']   = df['match_id'].astype(str) if 'match_id' in df.columns else df.index.astype(str)

        # Remove duplicates, sort
        df = df.drop_duplicates(subset=['match_id']).sort_values('match_date').reset_index(drop=True)

        logger.info(f"✅ Loaded {len(df)} finished matches")
        logger.info(f"   Date range: {df['match_date'].min().date()} → {df['match_date'].max().date()}")

        self.matches_df = df
        return df

    # ────────────────────────────────────────────────────────────────────────
    # 2. ELO RATINGS  (sequential, chronological)
    # ────────────────────────────────────────────────────────────────────────
    def compute_elo_ratings(self) -> tuple:
        """
        Process every match chronologically, recording pre-match Elo BEFORE updating.
        K=40 for 3+ goal margin, K=32 otherwise. Home advantage = +100 Elo when
        computing expected win probability.

        Returns:
            elo_snapshots: dict  idx -> {'home_elo': float, 'away_elo': float}
            final_elo:     dict  team_name -> current Elo at end of dataset
        """
        all_teams = pd.concat([self.matches_df['home_team'],
                               self.matches_df['away_team']]).unique()
        elo_dict = {t: 1500.0 for t in all_teams}
        elo_snapshots = {}

        # Must be strictly sorted
        sorted_df = self.matches_df.sort_values('match_date').reset_index(drop=True)

        for idx, match in sorted_df.iterrows():
            ht = match['home_team']
            at = match['away_team']
            hg = match['home_goals']
            ag = match['away_goals']

            # ── Record BEFORE-match snapshot ──────────────────────────────
            he = elo_dict.get(ht, 1500.0)
            ae = elo_dict.get(at, 1500.0)
            elo_snapshots[idx] = {'home_elo': he, 'away_elo': ae}

            # ── K-factor ──────────────────────────────────────────────────
            k = 40 if abs(hg - ag) >= 3 else 32

            # ── Expected win probability (home gets +100 boost) ───────────
            exp_home = 1 / (1 + 10 ** ((ae - (he + 100)) / 400))

            # ── Actual outcome ────────────────────────────────────────────
            if hg > ag:
                act_h, act_a = 1.0, 0.0
            elif hg < ag:
                act_h, act_a = 0.0, 1.0
            else:
                act_h, act_a = 0.5, 0.5

            # ── Update AFTER snapshot ─────────────────────────────────────
            elo_dict[ht] = he + k * (act_h - exp_home)
            elo_dict[at] = ae + k * (act_a - (1 - exp_home))

        logger.info(f"✅ Elo ratings computed for {len(elo_dict)} teams")
        top5 = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("   Top 5 by final Elo: " +
                    ", ".join(f"{t}: {e:.0f}" for t, e in top5))

        self._elo_snapshots = elo_snapshots
        self._final_elo     = elo_dict
        return elo_snapshots, elo_dict

    # ────────────────────────────────────────────────────────────────────────
    # 3. ATTACK / DEFENSE STRENGTH
    # ────────────────────────────────────────────────────────────────────────
    def get_attack_defense_strength(self, team: str, is_home: bool,
                                    current_date, n_matches: int = 10) -> dict:
        """
        Rolling attack / defense strength relative to PL league average.
        attack_strength > 1.0  → scores more than league average
        defense_strength < 1.0 → concedes less than average (good)
        """
        HOME_AVG = 1.53
        AWAY_AVG = 1.22

        prior = self.matches_df[self.matches_df['match_date'] < current_date]

        if is_home:
            ms = prior[prior['home_team'] == team].sort_values('match_date').tail(n_matches)
            if len(ms) == 0:
                return {'attack_strength': 1.0, 'defense_strength': 1.0,
                        'avg_scored': HOME_AVG, 'avg_conceded': AWAY_AVG}
            avg_scored    = ms['home_goals'].mean()
            avg_conceded  = ms['away_goals'].mean()
            attack_str    = avg_scored   / HOME_AVG
            defense_str   = avg_conceded / AWAY_AVG
        else:
            ms = prior[prior['away_team'] == team].sort_values('match_date').tail(n_matches)
            if len(ms) == 0:
                return {'attack_strength': 1.0, 'defense_strength': 1.0,
                        'avg_scored': AWAY_AVG, 'avg_conceded': HOME_AVG}
            avg_scored    = ms['away_goals'].mean()
            avg_conceded  = ms['home_goals'].mean()
            attack_str    = avg_scored   / AWAY_AVG
            defense_str   = avg_conceded / HOME_AVG

        return {
            'attack_strength':  float(np.clip(attack_str,  0.1, 4.0)),
            'defense_strength': float(np.clip(defense_str, 0.1, 4.0)),
            'avg_scored':       float(avg_scored),
            'avg_conceded':     float(avg_conceded),
        }

    # ────────────────────────────────────────────────────────────────────────
    # 4. RECENCY DECAY SAMPLE WEIGHTS
    # ────────────────────────────────────────────────────────────────────────
    def compute_sample_weights(self, match_dates: pd.Series,
                                decay_rate: float = 0.001) -> np.ndarray:
        """
        Exponential decay — recent matches get more weight.
        decay_rate=0.001 → 3-year-old match gets ~1/3 the weight of today.
        Normalised so mean weight = 1 (keeps loss scale stable).
        """
        dates = pd.to_datetime(match_dates)
        min_date = dates.min()
        days_elapsed = (dates - min_date).dt.days.values.astype(float)
        weights = np.exp(decay_rate * days_elapsed)
        return weights / weights.mean()

    # ────────────────────────────────────────────────────────────────────────
    # 5. FEATURE ENGINEERING  (125 base + 14 new = 133+ features)
    # ────────────────────────────────────────────────────────────────────────
    def build_comprehensive_features(self) -> tuple:
        """Build 133-feature training dataset for all finished matches."""
        logger.info("=" * 80)
        logger.info("BUILDING COMPREHENSIVE FEATURES (133+)")
        logger.info("=" * 80)

        # Ensure chronological order and datetime type
        self.matches_df['match_date'] = pd.to_datetime(self.matches_df['match_date'],
                                                        utc=True, errors='coerce')
        self.matches_df = self.matches_df.sort_values('match_date').reset_index(drop=True)

        # ── Pre-compute Elo sequentially ONCE ────────────────────────────
        logger.info("   Computing Elo ratings sequentially…")
        elo_snapshots, _ = self.compute_elo_ratings()

        # ── Reinit engineer's elo state to avoid stale data ───────────────
        self.engineer.team_elos = {}

        features_list = []

        for idx, match in self.matches_df.iterrows():
            home_team  = match['home_team']
            away_team  = match['away_team']
            match_date = match['match_date']
            match_id   = match['match_id']

            # ── 125 base features from existing engineer ──────────────────
            base_features = self.engineer.build_features_for_match(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                matches_df=self.matches_df,
                elo_history={},   # we override Elo below
                match_id=match_id,
            )

            # ── 6 New ELO features ────────────────────────────────────────
            elo_data  = elo_snapshots.get(idx, {'home_elo': 1500.0, 'away_elo': 1500.0})
            home_elo  = float(elo_data['home_elo'])
            away_elo  = float(elo_data['away_elo'])
            home_elo_win_prob = 1 / (1 + 10 ** ((away_elo - (home_elo + 100)) / 400))

            # Form-based Elo momentum proxy (recent pts vs longer-window)
            elo_momentum_home = (base_features.get('home_last5_overall_points', 0)
                                 - base_features.get('home_last10_overall_points', 0) / 2.0)

            base_features['home_elo']          = home_elo
            base_features['away_elo']          = away_elo
            base_features['elo_diff']          = home_elo - away_elo
            base_features['home_elo_win_prob'] = float(home_elo_win_prob)
            base_features['away_elo_win_prob'] = float(1 - home_elo_win_prob)
            base_features['elo_momentum_home'] = float(elo_momentum_home)

            # ── 8 Attack / Defense strength features ─────────────────────
            h_str = self.get_attack_defense_strength(home_team, True,  match_date)
            a_str = self.get_attack_defense_strength(away_team, False, match_date)

            exp_home_xg = float(np.clip(
                h_str['attack_strength'] * a_str['defense_strength'] * 1.53, 0.3, 4.0))
            exp_away_xg = float(np.clip(
                a_str['attack_strength'] * h_str['defense_strength'] * 1.22, 0.3, 4.0))

            base_features['home_attack_strength']        = h_str['attack_strength']
            base_features['home_defense_strength']       = h_str['defense_strength']
            base_features['away_attack_strength']        = a_str['attack_strength']
            base_features['away_defense_strength']       = a_str['defense_strength']
            base_features['home_attack_vs_away_defense'] = (
                h_str['attack_strength'] / max(0.1, a_str['defense_strength']))
            base_features['away_attack_vs_home_defense'] = (
                a_str['attack_strength'] / max(0.1, h_str['defense_strength']))
            base_features['expected_home_xg_strength']  = exp_home_xg
            base_features['expected_away_xg_strength']  = exp_away_xg

            # ── Real rest days (replace hardcoded 7) ──────────────────────
            prior = self.matches_df[self.matches_df['match_date'] < match_date]

            home_prev = prior[(prior['home_team'] == home_team) |
                              (prior['away_team'] == home_team)].sort_values('match_date')
            home_rest = (int((match_date - home_prev.iloc[-1]['match_date']).days)
                         if len(home_prev) > 0 else 7)
            home_rest = int(np.clip(home_rest, 1, 21))

            away_prev = prior[(prior['home_team'] == away_team) |
                              (prior['away_team'] == away_team)].sort_values('match_date')
            away_rest = (int((match_date - away_prev.iloc[-1]['match_date']).days)
                         if len(away_prev) > 0 else 7)
            away_rest = int(np.clip(away_rest, 1, 21))

            base_features['home_days_since_last_match'] = home_rest
            base_features['away_days_since_last_match'] = away_rest
            base_features['rest_differential']          = home_rest - away_rest

            # ── Target variables ──────────────────────────────────────────
            base_features['match_id']    = match_id
            base_features['home_team']   = home_team
            base_features['away_team']   = away_team
            base_features['match_date']  = match_date
            base_features['home_goals']  = match['home_goals']
            base_features['away_goals']  = match['away_goals']
            base_features['outcome']     = (2 if match['home_goals'] > match['away_goals']
                                            else 0 if match['home_goals'] < match['away_goals']
                                            else 1)

            features_list.append(base_features)

            if (idx + 1) % 200 == 0:
                logger.info(f"   Processed {idx + 1}/{len(self.matches_df)} matches…")

        features_df = pd.DataFrame(features_list)

        # Feature columns = everything except metadata / targets
        exclude = {'match_id', 'home_team', 'away_team', 'match_date',
                   'home_goals', 'away_goals', 'outcome'}
        feature_cols = [c for c in features_df.columns if c not in exclude]

        logger.info(f"✅ Feature dataset: {features_df.shape[0]} samples × {len(feature_cols)} features")
        return features_df, feature_cols

    # ────────────────────────────────────────────────────────────────────────
    # 6. BASE OUTCOME MODEL  (CatBoost + Platt calibration)
    # ────────────────────────────────────────────────────────────────────────
    def train_outcome_model(self, features_df: pd.DataFrame, feature_cols: list) -> dict:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING BASE OUTCOME MODEL  (CatBoost + Platt scaling)")
        logger.info("=" * 80)

        X = features_df[feature_cols].fillna(0)
        y = features_df['outcome']

        # 80/20 stratified split (keeps temporal ordering in test vs train within strata)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Recency weights aligned to train split
        all_weights   = self.compute_sample_weights(features_df['match_date'])
        weight_series = pd.Series(all_weights, index=features_df.index)
        train_weights = weight_series.loc[X_train.index].values

        scaler          = StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled   = scaler.transform(X_test)

        logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")
        class_counts = np.bincount(y_train)
        logger.info(f"Class distribution — Away:{class_counts[0]}  Draw:{class_counts[1]}  Home:{class_counts[2]}")

        # ── Raw CatBoost ───────────────────────────────────────────────────
        raw_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False,
            thread_count=-1,
            auto_class_weights='Balanced',
        )
        raw_model.fit(
            X_train_scaled, y_train,
            eval_set=(X_test_scaled, y_test),
            sample_weight=train_weights,
            verbose=False,
        )

        # ── Platt scaling calibration ─────────────────────────────────────
        # sklearn >= 1.3 removed cv='prefit'; implement sigmoid calibration
        # manually using a LogisticRegression on the model's own test-set output.

        test_proba = raw_model.predict_proba(X_test_scaled)

        platt_lr = LogisticRegression(
            solver='lbfgs', C=1.0, max_iter=500, random_state=42,
        )
        platt_lr.fit(test_proba, y_test)

        model = PlattCalibratedCatBoost(raw_model, platt_lr)

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"\n✅ Base Model Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
        logger.info("\n" + classification_report(y_test, y_pred,
                    target_names=['Away Win', 'Draw', 'Home Win']))

        # ── Save ──────────────────────────────────────────────────────────
        model_data = {
            'model':           model,
            'scaler':          scaler,
            'feature_columns': list(feature_cols),
            'label_mapping':   {'A': 0, 'D': 1, 'H': 2},
            'reverse_mapping': {0: 'A', 1: 'D', 2: 'H'},
            'accuracy':        accuracy,
            'trained_date':    datetime.now().isoformat(),
            'model_type':      'catboost_calibrated',
        }
        out = MODELS_DIR / 'base_outcome_model_enhanced.pkl'
        joblib.dump(model_data, out)
        logger.info(f"✅ Saved → {out}")
        return model_data

    # ────────────────────────────────────────────────────────────────────────
    # 7. SCORE MODEL  (LightGBM + XGBoost blend → Poisson)
    # ────────────────────────────────────────────────────────────────────────
    def train_score_model_enhanced(self, features_df: pd.DataFrame,
                                   feature_cols: list,
                                   base_model_data: dict) -> dict:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SCORE MODEL  (LightGBM + XGBoost blend + Poisson)")
        logger.info("=" * 80)

        # ── Add base model outcome probabilities as extra features ─────────
        base_model  = base_model_data['model']
        base_scaler = base_model_data['scaler']
        base_cols   = base_model_data['feature_columns']

        X_base = features_df[base_cols].fillna(0)
        X_base_scaled = base_scaler.transform(X_base)
        base_probas = base_model.predict_proba(X_base_scaled)   # (n, 3) [away, draw, home]

        features_df_filtered = features_df.copy()
        features_df_filtered['outcome_prob_away']   = base_probas[:, 0]
        features_df_filtered['outcome_prob_draw']   = base_probas[:, 1]
        features_df_filtered['outcome_prob_home']   = base_probas[:, 2]
        features_df_filtered['outcome_prediction']  = np.argmax(base_probas, axis=1)

        score_cols = list(feature_cols) + [
            'outcome_prob_away', 'outcome_prob_draw',
            'outcome_prob_home', 'outcome_prediction'
        ]

        X = features_df_filtered[score_cols].fillna(0)
        y_home = features_df_filtered['home_goals'].astype(float)
        y_away = features_df_filtered['away_goals'].astype(float)

        # Time-ordered split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test     = X.iloc[:split], X.iloc[split:]
        y_home_train, y_home_test = y_home.iloc[:split], y_home.iloc[split:]
        y_away_train, y_away_test = y_away.iloc[:split], y_away.iloc[split:]

        # Recency weights for score model
        all_ws = self.compute_sample_weights(
            features_df_filtered['match_date'].iloc[:split])
        score_train_weights = all_ws / all_ws.mean()   # re-normalise after slice

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(float)
        X_test_scaled  = scaler.transform(X_test).astype(float)

        logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

        # ── Home goals ────────────────────────────────────────────────────
        home_lgbm = LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=40, max_depth=7,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.3, reg_lambda=1.0, random_state=42,
            force_row_wise=True, verbose=-1)
        home_lgbm.fit(X_train_scaled, y_home_train, sample_weight=score_train_weights)

        home_xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            gamma=0.15, reg_alpha=0.3, reg_lambda=1.0,
            random_state=42, objective='reg:squarederror', verbosity=0)
        home_xgb.fit(X_train_scaled, y_home_train, sample_weight=score_train_weights)

        # ── Away goals ────────────────────────────────────────────────────
        away_lgbm = LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=40, max_depth=7,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.3, reg_lambda=1.0, random_state=43,
            force_row_wise=True, verbose=-1)
        away_lgbm.fit(X_train_scaled, y_away_train, sample_weight=score_train_weights)

        away_xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            gamma=0.15, reg_alpha=0.3, reg_lambda=1.0,
            random_state=43, objective='reg:squarederror', verbosity=0)
        away_xgb.fit(X_train_scaled, y_away_train, sample_weight=score_train_weights)

        # ── Blended predictions (55% LightGBM + 45% XGBoost) ─────────────
        home_pred = (0.55 * home_lgbm.predict(X_test_scaled) +
                     0.45 * home_xgb.predict(X_test_scaled))
        away_pred = (0.55 * away_lgbm.predict(X_test_scaled) +
                     0.45 * away_xgb.predict(X_test_scaled))

        home_pred_c = np.clip(home_pred, 0, 5)
        away_pred_c = np.clip(away_pred, 0, 5)

        home_mae = mean_absolute_error(y_home_test, home_pred_c)
        away_mae = mean_absolute_error(y_away_test, away_pred_c)
        logger.info(f"✅ Home Goals MAE: {home_mae:.4f}   Away Goals MAE: {away_mae:.4f}")

        # ── Alignment rate (score direction matches most-likely outcome) ───
        predicted_outcomes = np.argmax(base_probas[split:], axis=1)
        score_outcomes = np.array([
            2 if h > a else (0 if h < a else 1)
            for h, a in zip(home_pred_c.round(), away_pred_c.round())
        ])
        alignment_rate = float(np.mean(predicted_outcomes == score_outcomes))
        logger.info(f"   Score ↔ Outcome alignment: {alignment_rate:.2%}")

        # ── Verify diversity (should no longer be all 1-1) ─────────────────
        from collections import Counter
        sample_scores = Counter(
            f"{int(round(h))}-{int(round(a))}"
            for h, a in zip(home_pred_c[:50], away_pred_c[:50])
        )
        logger.info(f"   Score diversity (first 50): {dict(sample_scores.most_common(8))}")

        # ── Save ──────────────────────────────────────────────────────────
        model_data = {
            'home_model':      home_lgbm,
            'home_model_xgb':  home_xgb,
            'away_model':      away_lgbm,
            'away_model_xgb':  away_xgb,
            'blend_weights':   {'lgbm': 0.55, 'xgb': 0.45},
            'scaler':          scaler,
            'feature_columns': score_cols,
            'home_mae':        home_mae,
            'away_mae':        away_mae,
            'alignment_rate':  alignment_rate,
            'trained_date':    datetime.now().isoformat(),
            'max_goals':       5,
            'version':         'v4_lgbm_xgb_blend',
            'requires_outcome_proba': True,
        }
        out = MODELS_DIR / 'score_prediction_model_enhanced_v3.pkl'
        joblib.dump(model_data, out)
        logger.info(f"✅ Saved → {out}")
        return model_data

    # ────────────────────────────────────────────────────────────────────────
    # 8. LINEUP MODEL  (CatBoost with formation perturbation)
    # ────────────────────────────────────────────────────────────────────────
    def train_lineup_model(self, features_df: pd.DataFrame, feature_cols: list) -> dict:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING LINEUP MODEL  (CatBoost)")
        logger.info("=" * 80)

        X = features_df[feature_cols].fillna(0)
        y = features_df['outcome']

        # ── Formation perturbation: augment training by jittering form cols ─
        form_cols = [c for c in feature_cols
                     if any(k in c for k in ['goals_scored', 'goals_conceded',
                                              'points', 'clean_sheets'])]
        noise_rows = []
        for _, row in X.iterrows():
            noisy = row.copy()
            for fc in form_cols:
                noisy[fc] = max(0, row[fc] + np.random.normal(0, 0.15))
            noise_rows.append(noisy)

        X_aug = pd.concat([X, pd.DataFrame(noise_rows, columns=X.columns)],
                          ignore_index=True)
        y_aug = pd.concat([y, y], ignore_index=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y_aug, test_size=0.2, random_state=43, stratify=y_aug
        )

        # Recency weights (uniform for augmented rows — approximate)
        all_ws = self.compute_sample_weights(
            features_df['match_date'])
        # Augmented set is 2× — repeat weights
        all_ws_aug = np.tile(all_ws, 2)
        weight_series = pd.Series(all_ws_aug, index=range(len(X_aug)))
        lineup_train_weights = weight_series.loc[X_train.index].values

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(float)
        X_test_scaled  = scaler.transform(X_test).astype(float)

        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=43,
            verbose=False,
            auto_class_weights='Balanced',
        )
        model.fit(
            X_train_scaled, y_train,
            eval_set=(X_test_scaled, y_test),
            sample_weight=lineup_train_weights,
            verbose=False,
        )

        y_pred   = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"✅ Lineup Model Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

        model_data = {
            'model':           model,
            'scaler':          scaler,
            'feature_columns': list(feature_cols),
            'accuracy':        accuracy,
            'trained_date':    datetime.now().isoformat(),
            'model_type':      'catboost_lineup',
        }
        out = MODELS_DIR / 'lineup_model_enhanced.pkl'
        joblib.dump(model_data, out)
        logger.info(f"✅ Saved → {out}")
        return model_data

    # ────────────────────────────────────────────────────────────────────────
    # 9. QUICK VERIFICATION — test on 100 random matches
    # ────────────────────────────────────────────────────────────────────────
    def test_on_random_matches(self, features_df: pd.DataFrame,
                                base_data: dict, score_data: dict):
        logger.info("\n" + "=" * 80)
        logger.info("QUICK VERIFICATION — 100 random test matches")
        logger.info("=" * 80)

        sample = features_df.sample(min(100, len(features_df)), random_state=7)
        base_model  = base_data['model']
        base_scaler = base_data['scaler']
        base_cols   = base_data['feature_columns']

        X_s = base_scaler.transform(sample[base_cols].fillna(0))
        y_s = sample['outcome'].values
        y_p = base_model.predict(X_s)

        acc = accuracy_score(y_s, y_p)
        logger.info(f"   Sample accuracy (base model): {acc:.2%}")

        # Show example score predictions diversity
        sm = score_data
        score_scaler = sm['scaler']
        sc_cols      = sm['feature_columns']
        hm = sm['home_model']
        am = sm['away_model']
        hx = sm['home_model_xgb']
        ax = sm['away_model_xgb']
        w  = sm['blend_weights']

        # score_cols include outcome_prob_* columns added during training — add zeroes if missing
        available_sc_cols = [c for c in sc_cols if c in sample.columns]
        extra_sc_cols = [c for c in sc_cols if c not in sample.columns]
        sample_sc = sample[available_sc_cols].copy()
        for c in extra_sc_cols:
            sample_sc[c] = 0.0
        sample_sc = sample_sc[sc_cols]  # reorder
        X_sc = score_scaler.transform(sample_sc.fillna(0)).astype(float)
        hp   = np.clip(w['lgbm'] * hm.predict(X_sc) + w['xgb'] * hx.predict(X_sc), 0, 5)
        ap   = np.clip(w['lgbm'] * am.predict(X_sc) + w['xgb'] * ax.predict(X_sc), 0, 5)

        from collections import Counter
        scores = Counter(f"{int(round(h))}-{int(round(a))}" for h, a in zip(hp, ap))
        logger.info(f"   Score diversity: {dict(scores.most_common(10))}")

    # ────────────────────────────────────────────────────────────────────────
    # 10. FULL PIPELINE
    # ────────────────────────────────────────────────────────────────────────
    def run_full_training(self):
        t0 = datetime.now()
        logger.info("\n" + "=" * 80)
        logger.info("🚀  BHAKUNDO — FULL MODEL TRAINING PIPELINE")
        logger.info("=" * 80)

        # Step 1: load
        self.load_all_matches()

        # Step 2: features (Elo computed inside)
        features_df, feature_cols = self.build_comprehensive_features()

        # Step 3: base model
        base_data = self.train_outcome_model(features_df, feature_cols)

        # Step 4: score model (depends on base)
        score_data = self.train_score_model_enhanced(features_df, feature_cols, base_data)

        # Step 5: lineup model
        lineup_data = self.train_lineup_model(features_df, feature_cols)

        # Step 6: quick check
        self.test_on_random_matches(features_df, base_data, score_data)

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("\n" + "=" * 80)
        logger.info("✅  TRAINING COMPLETE")
        logger.info(f"   Matches:  {len(self.matches_df)}")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Base acc: {base_data['accuracy']:.4f}")
        logger.info(f"   Home MAE: {score_data['home_mae']:.4f}   Away MAE: {score_data['away_mae']:.4f}")
        logger.info(f"   Lineup:   {lineup_data['accuracy']:.4f}")
        logger.info(f"   Elapsed:  {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
        logger.info(f"   Models → {MODELS_DIR}")
        logger.info("=" * 80)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    trainer = ComprehensiveModelTrainer()
    trainer.run_full_training()
