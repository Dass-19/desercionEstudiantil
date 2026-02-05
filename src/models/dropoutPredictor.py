import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
import seaborn as sns
from src.utils.features import featureEngineering


class DropoutPredictor:
    """
    Dropout prediction system using a trained imbalanced-learn pipeline.
    """

    def __init__(self, artifact_paths: list[str]):
        rf_artifact = joblib.load(artifact_paths[0])

        self.rf_model = rf_artifact["model"]
        self.rf_threshold = rf_artifact["threshold"]
        self.rf_features = rf_artifact["features"]
        self.rf_version = rf_artifact["model_version"]

        if hasattr(self.rf_model, 'named_steps'):
            model_step = self.rf_model.steps[-1][1]
        else:
            model_step = self.rf_model

        self.explainer = shap.TreeExplainer(model_step)
        self.data = None

        lr_artifact = joblib.load(artifact_paths[1])
        self.lr_model = lr_artifact["model"]
        self.lr_threshold = lr_artifact["threshold"]
        self.lr_features = lr_artifact["features"]
        self.lr_version = lr_artifact["model_version"]

    def predict(self, X: pd.DataFrame):
        '''
        Predicts dropout probability and applies the stored decision threshold.

        :param X: Feature set for prediction.
        :type X: pd.DataFrame
        '''
        X = X.copy()
        X = featureEngineering(X)
        self.data = X

        rf_proba = self.rf_model.predict_proba(X[self.rf_features])[0, 1]
        rf_pred = int(rf_proba >= self.rf_threshold)

        lr_proba = self.lr_model.predict_proba(X[self.lr_features])[0, 1]

        return rf_proba, rf_pred, lr_proba

    def explain(self):
        '''
        Returns a graph showing the impact of each variable on the prediction

        :param X: Feature set for prediction.
        :type X: pd.DataFrame
        '''
        if self.data is None:
            raise ValueError("Run predict() before explain().")

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        X_explain = self.data[self.rf_features].iloc[[0]]

        if hasattr(self.rf_model, 'named_steps'):
            X_transformed = X_explain
            for name, transformer in self.rf_model.steps[:-1]:
                X_transformed = transformer.transform(X_transformed)
            shap_values = self.explainer(X_transformed)
        else:
            shap_values = self.explainer(X_explain)

        if hasattr(shap_values, 'values'):
            sv = shap_values.values
        else:
            sv = shap_values

        sv = np.asarray(sv)

        if sv.ndim == 3:
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            sv = sv[0, :]

        sv = sv.flatten()


        assert len(sv) == len(self.rf_features), (
            f"SHAP values length ({len(sv)}) != features length ({len(self.rf_features)})"
        )

        df_expl = pd.DataFrame({
            "Variable": self.rf_features,
            "Impacto": sv
        }).sort_values(
            by="Impacto", key=np.abs, ascending=True
        )

        df_expl["Tipo"] = np.where(
            df_expl["Impacto"] > 0,
            "Aumenta riesgo",
            "Disminuye riesgo"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {
            "Aumenta riesgo": "#E74C3C",
            "Disminuye riesgo": "#27AE60"
        }

        sns.barplot(
            data=df_expl,
            y="Variable",
            x="Impacto",
            hue="Tipo",
            palette=palette,
            ax=ax,
            dodge=False,
            alpha=0.85
        )

        ax.axvline(0, color="#34495E", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="Efecto")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        return fig
