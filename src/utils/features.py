import numpy as np
import pandas as pd


def featureEngineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering over the student dataset to create
    derived academic performance and risk indicators.

    This function generates continuous and binary features related to
    academic efficiency, attendance, academic level, and combined risk
    factors, without modifying the original DataFrame.

    Engineered features:
    - TASA_REPROB: Ratio of failed subjects to total enrolled subjects.
    - EFICIENCIA: Attendance-weighted academic efficiency.
    - ASIST_DEFICIENTE: Binary flag for low attendance (< 70%).
    - PROM_BAJO: Binary flag for low academic average (< 7).
    - RIESGO_ASIST_Y_PROM: Combined risk indicator for low attendance
      and low academic performance.
    - RIESGO_NIVEL_BAJO: Binary flag for early academic levels (<= 3).
    - PROM_X_NIVEL: Interaction feature between academic average and
      academic level using logarithmic scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing academic and attendance-related features.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the engineered features added.
    """
    df = df.copy()

    df["PROP_REPROB"] = df["REPROBADAS"] / df["TOTAL_MAT"]

    df["TASA_APROB"] = (df['TOTAL_MAT'] - df['REPROBADAS']) / df['TOTAL_MAT']

    df['INDX_REPIT'] = df['REPITENCIAS'] / df['TOTAL_MAT']

    df['EFICIENCIA'] = df['ASIST_PROM'] * (1 - df['PROP_REPROB'])

    df['REPROB_x_ASIST'] = df['REPROBADAS'] * (100 - df['ASIST_PROM'])

    df["EFICIENCIA_x_CARGA"] = df["EFICIENCIA"] * df["TOTAL_MAT"]

    df["RIESGO_CRITICO"] = (
        (df["ASIST_PROM"] < 65) &
        (df["REPROBADAS"] >= 2)
        ).astype(int)

    df["REPIT_PERSISTENTE"] = (
        (df["REPITENCIAS"] > 0) &
        (df["REPROBADAS"] > 0)
        ).astype(int)

    return df
