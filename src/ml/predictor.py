"""
AssureX Claim Engine — Tabular ML Inference Service

Loads the serialized best model artifact and provides clean, fast prediction
and probability estimation for live warranty claim evaluations.
Supports both single claim inference and high-throughput batch inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from .preprocessing import clean_dataframe, INT_TO_LABEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.joblib")


class TabularPredictor:
    """Production inference wrapper for the best trained tabular model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model_artifact = None
        self.model = None
        self.preprocessor = None
        self.model_name = "Unloaded"
        self._load_model()

    def _load_model(self):
        """Load serialized model artifact from disk."""
        if not os.path.exists(self.model_path):
            print(f"[TabularPredictor] Warning: Model file not found at {self.model_path}")
            return

        self.model_artifact = joblib.load(self.model_path)
        if isinstance(self.model_artifact, dict) and "model" in self.model_artifact:
            self.model = self.model_artifact["model"]
            self.preprocessor = self.model_artifact.get("preprocessor")
            self.model_name = self.model_artifact.get("model_name", "BestModel")
        else:
            self.model = self.model_artifact
            self.model_name = type(self.model).__name__
            preprocessor_path = os.path.join(os.path.dirname(self.model_path), "preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)

        # Fallback preprocessor load if still missing
        if self.preprocessor is None:
            preprocessor_path = os.path.join(os.path.dirname(self.model_path), "preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)

        print(f"[TabularPredictor] Successfully loaded {self.model_name} from {self.model_path}")

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None and self.preprocessor is not None

    def predict_batch(self, claims_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        High-throughput batch prediction for multiple claim dictionaries.
        """
        if not self.is_ready:
            return [
                {
                    "model_type": "Python_Tabular",
                    "model_name": "Fallback_Rule_Estimator",
                    "predicted_class": "Manual Review Required",
                    "confidence_scores": {
                        "Likely Valid": 0.3333,
                        "Likely Invalid": 0.3333,
                        "Manual Review Required": 0.3334,
                    },
                    "top_confidence": 0.3334,
                }
                for _ in claims_data
            ]

        df = pd.DataFrame(claims_data)
        df_clean = clean_dataframe(df)
        X = self.preprocessor.transform(df_clean)

        # Get class probabilities
        if hasattr(self.model, "predict_proba"):
            all_probs = self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            import scipy.special
            df_vals = self.model.decision_function(X)
            all_probs = scipy.special.softmax(df_vals, axis=1)
        else:
            preds = self.model.predict(X)
            all_probs = np.zeros((len(preds), 3))
            for i, p in enumerate(preds):
                all_probs[i, p] = 1.0

        results = []
        for i in range(len(claims_data)):
            probs = all_probs[i]
            class_names = [INT_TO_LABEL[k] for k in range(len(probs))]
            confidence_dict = {
                class_names[k]: round(float(probs[k]), 4) for k in range(len(probs))
            }
            top_class_idx = int(probs.argmax())
            top_class = INT_TO_LABEL[top_class_idx]
            top_conf = round(float(probs[top_class_idx]), 4)

            results.append({
                "model_type": "Python_Tabular",
                "model_name": self.model_name,
                "predicted_class": top_class,
                "confidence_scores": confidence_dict,
                "top_confidence": top_conf,
            })
        return results

    def predict(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outcome and class probabilities for a single claim dictionary.
        """
        return self.predict_batch([claim_data])[0]
