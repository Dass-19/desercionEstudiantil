import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def logPredictionSupabase(
        probability_rf: float,
        probability_lr: float,
        prediction: int,
        threshold: float,
        model_version: str
        ):
    supabase = get_supabase_client()

    data = {
        "probability_rf": round(probability_rf, 4),
        "probability_lr": round(probability_lr, 4),
        "prediction": int(prediction),
        "threshold": threshold,
        "model_version": model_version
    }

    supabase.table("logs").insert(data).execute()
