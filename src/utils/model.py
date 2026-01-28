import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE


def tuneHyperparameters(
        X_train: pd.DataFrame,
        y_train: pd.Series
        ):
    '''
    Return the best hyperpparameters of a data umbalanced.

    We use Smote from `SMOTE` and `RandomForestClassifier`.

    :param X_train: Train data
    :type X_train: pd.DataFrame
    :param y_train: Train data
    :type y_train: pd.DataFrame
    '''

    pipe = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42,
                                      class_weight='balanced'))
    ])

    paramDist = {
        'smote__sampling_strategy': ['auto', 'not majority'],
        'smote__k_neighbors': [3, 5, 7],
        'rf__n_estimators': [100, 200, 300, 400, 500],
        'rf__max_depth': [6, 9, 12, 15, 18, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 3, 5],
        'rf__max_features': ['sqrt', 'log2', None],
        'rf__criterion': ['gini', 'entropy']
    }

    cv = TimeSeriesSplit(n_splits=3)

    scoring = {
        'balanced_accuracy': 'balanced_accuracy',
        'f1_macro': make_scorer(f1_score, average='macro'),
        'roc_auc': 'roc_auc'
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=paramDist,
        n_iter=40,
        cv=cv,
        scoring=scoring,
        refit='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=2)

    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print(f"Best F1 macro: {search.best_score_:.4f}")

    return search.best_estimator_


def findBestThreshold(y_test: pd.Series,
                      y_proba: pd.DataFrame,
                      min_recall=0.6) -> float:
    '''
    Finds the optimal decision threshold based on a minimum recall constraint
    for the positive class (dropout).

    The function evaluates different thresholds derived from the
    Precision-Recall curve and selects the one that maximizes the F1-score
    while ensuring a minimum recall level.

    :param y_test: True binary labels
    :type y_test: pd.Series
    :param y_proba: Predicted probabilities for the positive class.
    :type y_proba: pd.DataFrame
    :param min_recall: Minimum recall required for the positive class.

    Returns
    -------
    float
        Selected optimal decision threshold.
    '''
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision[:-1],
        'recall': recall[:-1]
    })

    df['f1'] = 2 * (df['precision'] * df['recall']) / (
        df['precision'] + df['recall'] + 1e-8
    )

    df_valid = df[df['recall'] >= min_recall]

    if df_valid.empty:
        return df.sort_values('f1', ascending=False).iloc[0]['threshold']

    return df_valid.sort_values('f1', ascending=False).iloc[0]['threshold']
