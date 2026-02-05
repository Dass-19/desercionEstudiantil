import numpy as np
import pandas as pd


def featureEngineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["TASA_REPROB"] = df["REPROBADAS"] / df["TOTAL_MAT"].clip(lower=1)

    df["TASA_APROB"] = 1 - df["TASA_REPROB"]

    df['EFICIENCIA'] = df['ASIST_PROM'] * df["TASA_APROB"]

    df["REPROB_X_MATERIA"] = df["REPROBADAS"] / df["TOTAL_MAT"].clip(lower=1)

    df["CARGA_ACADEMICA"] = df["TOTAL_MAT"]

    df["ASIST_DEFICIENTE"] = (df["ASIST_PROM"] < 70).astype(int)

    df["PROM_BAJO"] = (df["PROM_PERIODO"] < 7).astype(int)

    df["RIESGO_ASIST_Y_PROM"] = (
        (df["ASIST_DEFICIENTE"] == 1) &
        (df["PROM_BAJO"] == 1)
    ).astype(int)

    df["REPITENCIA_ALTA"] = (df["REPITENCIAS"] >= 2).astype(int)

    df["RIESGO_NIVEL_BAJO"] = (df["NIVEL"] <= 3).astype(int)

    df["PROM_X_NIVEL"] = df["PROM_PERIODO"] * np.log1p(df["NIVEL"])

    return df
