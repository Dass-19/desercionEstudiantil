import streamlit as st
from supabase import create_client, Client
import pandas as pd


@st.cache_resource
def get_supabase_client() -> Client:
    """
    Create and cache a Supabase client instance.

    The client is initialized using credentials stored in
    Streamlit secrets and cached to avoid re-creation on
    each function call.

    Returns
    -------
    supabase.Client
        Authenticated Supabase client instance.
    """
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def logPredictionSupabase(
        X: pd.DataFrame,
        probability_rf: float,
        probability_lr: float,
        prediction: int,
        threshold: float,
        model_version: str
        ):
    """
    Log a model prediction and related student metrics into Supabase.

    This function extracts the relevant student features from the
    input DataFrame, combines them with model outputs, and stores
    the information in the Supabase 'logs' table for monitoring
    and traceability purposes.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing a single student record with
        the required feature columns.
    probability_rf : float
        Predicted probability from the Random Forest model.
    probability_lr : float
        Predicted probability from the Logistic Regression model.
    prediction : int
        Final binary prediction (e.g., 0 = no dropout, 1 = dropout).
    threshold : float
        Decision threshold used to generate the final prediction.
    model_version : str
        Identifier of the model version used for the prediction.

    Returns
    -------
    None
        This function performs a database insert operation and
        does not return any value.
    """
    supabase = get_supabase_client()
    student_metrics = {
        "PROM_PERIODO": float(X['PROM_PERIODO'][0]),
        "ASIST_PROM": float(X['ASIST_PROM'][0]),
        "TOTAL_MAT": int(X['TOTAL_MAT'][0]),
        "REPROBADAS": int(X['REPROBADAS'][0]),
        "REPITENCIAS": int(X['REPITENCIAS'][0]),
        "NIVEL": int(X['NIVEL'][0]),
        "PROP_REPROB": float(X['PROP_REPROB'][0]),
        "EFICIENCIA": float(X['EFICIENCIA'][0])
    }

    data = {
        "probability_rf": round(probability_rf, 4),
        "probability_lr": round(probability_lr, 4),
        "prediction": int(prediction),
        "threshold": threshold,
        "model_version": model_version,
        "student_data": student_metrics
    }

    supabase.table("logs").insert(data).execute()
