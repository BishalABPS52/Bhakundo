"""Data preprocessing and cleaning pipeline for Premier League predictions"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning, preprocessing, and splitting"""
    
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        """
        Initialize preprocessor
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and outliers
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Cleaning data: {len(df)} rows, {len(df.columns)} columns")
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # 1. Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # 2. Handle missing values
        df = self._handle_missing_values(df)
        
        # 3. Handle outliers
        df = self._handle_outliers(df)
        
        # 4. Validate data types
        df = self._validate_data_types(df)
        
        logger.info(f"Data cleaning complete: {len(df)} rows remaining")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Handling missing values in {len(missing_cols)} columns")
        
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            # If more than 50% missing, drop the column
            if missing_pct > 50:
                logger.warning(f"Dropping {col}: {missing_pct:.1f}% missing")
                df = df.drop(columns=[col])
                continue
            
            # For numeric columns: fill with median (robust to outliers)
            if df[col].dtype in ['int64', 'float64']:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled {col} missing values with median: {median_value:.2f}")
            
            # For categorical columns: fill with mode or 'Unknown'
            else:
                if df[col].mode().empty:
                    df[col] = df[col].fillna('Unknown')
                else:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method for numeric features
        Caps outliers at 1.5 * IQR beyond Q1 and Q3
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers instead of removing them (preserves data)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outliers} outliers in {col}")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data types are appropriate"""
        
        # Convert object columns that should be numeric
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
                logger.info(f"Converted {col} to numeric")
            except (ValueError, TypeError):
                pass  # Keep as object/string
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            Dataframe with encoded features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            logger.info("No categorical features to encode")
            return df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical features")
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # Use label encoding for binary categories
            unique_vals = df[col].nunique()
            
            if unique_vals == 2:
                # Binary encoding
                if fit:
                    df[col] = df[col].astype('category').cat.codes
                else:
                    df[col] = df[col].astype('category').cat.codes
                logger.info(f"Label encoded {col} (binary)")
            
            elif unique_vals <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                logger.info(f"One-hot encoded {col} ({unique_vals} categories)")
            
            else:
                # Label encoding for high cardinality
                if fit:
                    df[col] = df[col].astype('category').cat.codes
                else:
                    df[col] = df[col].astype('category').cat.codes
                logger.info(f"Label encoded {col} ({unique_vals} categories)")
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numeric features using StandardScaler (z-score normalization)
        
        Args:
            X: Feature dataframe
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted and transformed {X.shape[1]} features")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info(f"Transformed {X.shape[1]} features")
        
        return X_scaled
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        temporal_split: bool = False,
        date_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target variable
            temporal_split: If True, use temporal ordering instead of random split
            date_column: Column name for temporal sorting (if temporal_split=True)
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Splitting data: {len(X)} samples")
        
        if temporal_split and date_column is not None:
            # Temporal split: train on older data, test on newer data
            logger.info("Using temporal split (simulates real-world scenario)")
            
            # Sort by date
            if date_column in X.columns:
                sort_idx = X[date_column].argsort()
                X = X.iloc[sort_idx]
                y = y.iloc[sort_idx]
            
            # Split based on time
            n_samples = len(X)
            test_idx = int(n_samples * (1 - self.test_size))
            val_idx = int(test_idx * (1 - self.val_size))
            
            X_train = X.iloc[:val_idx]
            X_val = X.iloc[val_idx:test_idx]
            X_test = X.iloc[test_idx:]
            
            y_train = y.iloc[:val_idx]
            y_val = y.iloc[val_idx:test_idx]
            y_test = y.iloc[test_idx:]
            
        else:
            # Random split with stratification
            logger.info("Using random stratified split")
            
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if y.dtype == 'object' or y.nunique() < 10 else None
            )
            
            # Second split: train vs val
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=y_temp if y_temp.dtype == 'object' or y_temp.nunique() < 10 else None
            )
        
        logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Val set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check for class imbalance
        if y.dtype == 'object' or y.nunique() < 10:
            logger.info("\nClass distribution:")
            logger.info(f"Train: {dict(y_train.value_counts())}")
            logger.info(f"Val: {dict(y_val.value_counts())}")
            logger.info(f"Test: {dict(y_test.value_counts())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        exclude_columns: Optional[List[str]] = None,
        temporal_split: bool = True,
        date_column: Optional[str] = None
    ) -> Dict:
        """
        Complete preprocessing pipeline: clean, encode, scale, and split
        
        Args:
            df: Raw dataframe
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            temporal_split: Use temporal ordering for split
            date_column: Date column for temporal split
            
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive data preprocessing pipeline")
        logger.info("=" * 60)
        
        # 1. Clean data
        df_clean = self.clean_data(df)
        
        # 2. Separate features and target
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df_clean[target_column]
        
        # Exclude specified columns and target
        exclude = [target_column] + (exclude_columns or [])
        X = df_clean.drop(columns=exclude, errors='ignore')
        
        # Store date column if temporal split
        date_series = None
        if temporal_split and date_column and date_column in X.columns:
            date_series = X[date_column].copy()
            # Don't drop date column yet, needed for sorting
        
        logger.info(f"Features: {X.shape[1]} columns, Target: {target_column}")
        
        # 3. Encode categorical features
        X_encoded = self.encode_features(X, fit=True)
        
        # Remove date column after encoding if it exists
        if date_column and date_column in X_encoded.columns:
            X_encoded = X_encoded.drop(columns=[date_column])
        
        # 4. Store feature names
        self.feature_names = X_encoded.columns.tolist()
        logger.info(f"Final features after encoding: {len(self.feature_names)}")
        
        # 5. Split data (before scaling to prevent data leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_encoded, y, 
            temporal_split=temporal_split,
            date_column=None  # Already sorted if needed
        )
        
        # 6. Scale features (fit only on training data)
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        logger.info("=" * 60)
        logger.info("Data preprocessing complete!")
        logger.info("=" * 60)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'n_samples': len(df_clean),
            'n_features': len(self.feature_names)
        }


def get_evaluation_metrics() -> Dict:
    """
    Define comprehensive evaluation metrics for model assessment
    
    Returns:
        Dictionary of metric names and functions
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score
    )
    
    classification_metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'log_loss': log_loss,
        'confusion_matrix': confusion_matrix
    }
    
    regression_metrics = {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score
    }
    
    return {
        'classification': classification_metrics,
        'regression': regression_metrics
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples) * 10 + 50,
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.randn(n_samples) * 5,
        'target': np.random.choice([0, 1, 2], n_samples)
    })
    
    # Add some missing values
    df.loc[df.sample(frac=0.1).index, 'feature1'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'feature3'] = np.nan
    
    # Preprocess
    preprocessor = DataPreprocessor(test_size=0.2, val_size=0.1)
    result = preprocessor.prepare_features(df, target_column='target')
    
    print(f"\nProcessed data shape:")
    print(f"Train: {result['X_train'].shape}")
    print(f"Val: {result['X_val'].shape}")
    print(f"Test: {result['X_test'].shape}")
    print(f"\nFeature names: {result['feature_names']}")
