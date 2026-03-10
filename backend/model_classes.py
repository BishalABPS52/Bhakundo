"""
Shared model wrapper classes — must be importable at pickle load time.
Both the training script and all inference scripts should
`from backend.model_classes import PlattCalibratedCatBoost`
"""
import numpy as np


class PlattCalibratedCatBoost:
    """
    CatBoost base model wrapped with a Platt-scaling LogisticRegression.
    Defined in a standalone importable module so joblib/pickle can
    deserialise it regardless of which script loads the model.
    """

    def __init__(self, base, platt):
        self.base    = base
        self.platt   = platt
        self.classes_ = base.classes_

    def predict_proba(self, X):
        return self.platt.predict_proba(self.base.predict_proba(X))

    def predict(self, X):
        return self.platt.predict(self.base.predict_proba(X))
