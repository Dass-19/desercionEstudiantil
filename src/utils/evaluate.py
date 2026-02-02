from pathlib import Path
import sys
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    roc_curve
)
from utils.model import findBestThreshold
from dotenv import load_dotenv


ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(ROOT))


load_dotenv()
METRICS_PATH = ROOT / str(os.getenv("METRICS_PATH"))
CM_PATH = ROOT / str(os.getenv("CM_PATH"))
FP_PATH = ROOT / str(os.getenv("FP_PATH"))


def modelEvaluation(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series):
    '''
    Returns `CV Accuracy`, `Accuracy`, `Balanced accuracy`,
    `Classification report`, `Feature importances`, 'ROC-AUC'
    and `Confussion matrix`.

    :param pipeline: Predictor pipeline
    :type pipeline: Pipeline
    :param X: Data
    :type X: pd.DataFrame
    :param y: Data
    :type y: pd.DataFrame
    :param X_test: Data
    :type X_test: pd.DataFrame
    :param y_test: Data
    :type y_test: pd.DataFrame
    '''
    if isinstance(pipeline, Pipeline):
        model = pipeline.steps[-1][1]
    else:
        model = pipeline

    cv = TimeSeriesSplit(n_splits=3)

    cv_auc = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring='roc_auc'
        ).mean()
    cv_acc = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring='accuracy'
        ).mean()

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    acc_score = accuracy_score(y_test, y_pred)
    balanced_acc_score = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precission = precision_score(y_test, y_pred, average='macro')

    print(f"CV ROC-AUC: {cv_auc * 100:.2f}%")
    print(f"CV Accuracy: {cv_acc * 100:.2f}%")
    print(f"Test ROC-AUC: {test_auc * 100:.2f}%")
    print(f"Test Accuracy: {acc_score * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc_score * 100:.2f}%")
    print("\nClassification Report:\n")
    present_classes = [str(c) for c in model.classes_ if c in y_test.unique()]
    print(classification_report(y_test, y_pred, target_names=present_classes))

    metrics = {
        "accuracy": acc_score,
        "roc_auc": test_auc,
        "balanced_accuracy": balanced_acc_score,
        "f1_score": f1,
        "recall": recall,
        "precission": precission
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12)
        )

    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame(
            {
                'Feature': pipeline.named_steps['rf'].feature_names_in_,
                'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

        importances.to_csv(FP_PATH, index=False)

        sns.barplot(
            x='Importance',
            y='Feature',
            data=importances.head(10),
            ax=axes[0, 0],
            color="skyblue"
            )
        axes[0, 0].set_title("Feature Importances")

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0, 1],
        xticklabels=model.classes_,
        yticklabels=model.classes_
        )
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('True')
    axes[0, 1].set_title('Confusion Matrix')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1, 0].plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'AUC = {test_auc:.2f}'
        )
    axes[1, 0].plot(
        [0, 1],
        [0, 1],
        color='navy',
        lw=2,
        linestyle='--'
        )
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Receiver Operating Characteristic (ROC)')
    axes[1, 0].legend(loc="lower right")

    pd.DataFrame(cm).to_csv(CM_PATH, index=False)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    plt.tight_layout()
    plt.show()


def modelEvaluationWithThreshold(
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        min_recall: float = 0.6
        ) -> float:
    """
    Evaluates a trained classification pipeline using a dynamically selected
    decision threshold optimized for recall of the positive class.

    The threshold is selected based on the Precision–Recall curve to ensure
    a minimum recall level, making the evaluation suitable for imbalanced
    problems such as student dropout prediction.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline with a classifier supporting predict_proba.
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target labels.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Test target labels.
    min_recall : float, optional (default=0.6)
        Minimum recall required for the positive class.

    Returns
    -------
    float
        Selected decision threshold.
    """

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    threshold = findBestThreshold(y_test, y_proba, min_recall)

    y_pred = (y_proba >= threshold).astype(int)

    print(f"Selected threshold: {threshold:.2f}\n")

    print(f"Recall (dropout): {recall_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Dynamic Threshold)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return threshold
