import streamlit as st
from supabase import create_client, Client
import pandas as pd


@st.cache_resource
def get_supabase_client() -> Client:
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
