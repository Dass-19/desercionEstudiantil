import pandas as pd
import joblib
from imblearn.pipeline import Pipeline


class DropoutPredictor:
    """
    Dropout prediction system using a trained imbalanced-learn pipeline
    and a custom decision threshold.
    """

    def __init__(self, model: Pipeline, threshold: float):
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: str) -> "DropoutPredictor":
        """
        Loads a trained dropout predictor from disk.

        Parameters
        ----------
        path : str
            Path to the serialized model artifact.

        Returns
        -------
        DropoutPredictor
            Loaded predictor instance.
        """
        artifact = joblib.load(path)
        return cls(
            model=artifact['model'],
            threshold=artifact['threshold']
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts dropout probability and applies the stored decision threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature set for prediction.

        Returns
        -------
        pd.DataFrame
            DataFrame with dropout probability and binary prediction.
        """
        proba = self.model.predict_proba(X)[:, 1]

        return pd.DataFrame({
            'dropout_probability': proba,
            'dropout_prediction': (proba >= self.threshold).astype(int)
        })
