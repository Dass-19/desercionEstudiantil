import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
import seaborn as sns


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
        model_final = self.rf_model[-1] if hasattr(self.rf_model, "steps") else self.model
        self.explainer = shap.TreeExplainer(model_final)
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

        X["PROP_REPROB"] = X["REPROBADAS"] / X["TOTAL_MAT"]
        X["EFICIENCIA"] = X["ASIST_PROM"] * (1 - X["PROP_REPROB"])

        rf_proba = self.rf_model.predict_proba(X[self.rf_features])[0, 1]
        rf_pred = int(rf_proba >= self.rf_threshold)

        self.data = X
        lr_proba = self.lr_model.predict_proba(X[self.lr_features])[0, 1]

        return rf_proba, rf_pred, lr_proba

    def explain(self):
        '''
        Returns a graph showing the impact of each variable on the prediction

        :param X: Feature set for prediction.
        :type X: pd.DataFrame
        '''
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        shap_values = self.explainer.shap_values(self.data)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        sv = np.asarray(sv)

        if sv.ndim == 3:
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            sv = sv[0, :]

        sv = sv.ravel()

        if sv.shape[0] != len(self.rf_features):
            if sv.shape[0] == 2 * len(self.rf_features):
                sv = sv[len(self.rf_features):]
            else:
                pass

        df_expl = pd.DataFrame({
            "Variable": self.rf_features,
            "Impacto": sv
        }).sort_values(by="Impacto", key=abs, ascending=True)

        df_expl['Tipo'] = df_expl['Impacto'].apply(
            lambda x: 'Aumenta riesgo' if x > 0 else 'Disminuye riesgo'
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        palette = {'Aumenta riesgo': '#E74C3C', 'Disminuye riesgo': '#27AE60'}

        sns.barplot(
            data=df_expl,
            y='Variable',
            x='Impacto',
            hue='Tipo',
            palette=palette,
            ax=ax,
            dodge=False,
            alpha=0.85
        )

        ax.axvline(
            x=0,
            color='#34495E',
            linestyle='-',
            linewidth=1.5,
            alpha=0.7
            )

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.legend(
            title='Efecto',
            title_fontsize=10,
            fontsize=9,
            loc='upper right',
            frameon=True,
            shadow=True
        )
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_axisbelow(True)

        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        return fig