import streamlit as st
import pandas as pd
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from models.dropoutPredictor import DropoutPredictor
from utils.figures import (
    plotConfusionMatrix,
    plotFeatureImportances,
    plotAttendanceVsMean,
    plotPromAverage,
    plotStudentsLevel
)


load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
METRICS_PATH = os.getenv("METRICS_PATH")
CM_PATH = os.getenv("CM_PATH")
FP_PATH = os.getenv("FP_PATH")
PROCESSED_DATA_PATH = os.getenv("ALL_PROCESSED_DATA_PATH")


st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    layout="wide")


@st.cache_resource
def load_model():
    try:
        dp = DropoutPredictor(f"../{MODEL_PATH}" if MODEL_PATH else "")
        return dp
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None


dropoutPredictor = load_model()


@st.cache_data
def load_metrics():
    with open(METRICS_PATH if METRICS_PATH else "") as f:
        return json.load(f)


metrics = load_metrics()


@st.cache_data
def get_cm_data(file_path):
    return pd.read_csv(file_path)


cm_data = get_cm_data(CM_PATH)


@st.cache_data
def get_fp_data(file_path):
    return pd.read_csv(file_path)


fp_data = get_fp_data(FP_PATH)


@st.cache_data
def load_data():
    df = pd.read_excel(f"../{PROCESSED_DATA_PATH}")
    return df


st.sidebar.title("Navegación")
seccion = st.sidebar.radio(
    "Secciones",
    [
        "Dashboard EDA",
        "Métricas del modelo",
        "Predicción individual"
        ])

if seccion == "Dashboard EDA":
    st.title("Análisis Exploratorio de Datos (EDA)")

    df = load_data()

    total_est = df['ESTUDIANTE'].nunique()
    tasa_riesgo = df['RIESGO_t1'].mean()
    prom_gral = df['PROM_PERIODO'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total estudiantes", f"{total_est}")
    col2.metric("Tasa de deserción", f"{tasa_riesgo:.2%}")
    col3.metric("Promedio general", f"{prom_gral:.2f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribución de notas", text_alignment='center')
        st.pyplot(
            plotPromAverage(df),
            width='stretch'
            )

    with col_right:
        st.subheader("Estudiantes por nivel", text_alignment='center')
        st.pyplot(
            plotStudentsLevel(df),
            width='stretch'
            )

    st.subheader("Relación asistencia vs rendimiento", text_alignment='center')
    st.pyplot(
        plotAttendanceVsMean(df),
        width='stretch'
        )

    with st.expander("Ver tabla de datos"):
        st.dataframe(
            df.head(15),
            width='stretch'
            )

elif seccion == "Métricas del modelo":
    st.title("Evaluación del modelo")
    st.caption("Algoritmo: Random Forest Classifier")

    with st.container(border=True):
        cols = st.columns(3)
        cols[0].metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
            help="Precisión global del modelo"
            )
        cols[1].metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.2%}",
            help="Capacidad de discriminación entre clases"
            )
        cols[2].metric(
            "Balanced Acc",
            f"{metrics['balanced_accuracy']:.2%}",
            help="Precisión ajustada por desbalance de clases"
            )

    tab1, tab2 = st.tabs([
        "Diagnóstico de errores",
        "Interpretación del modelo"
        ])

    with tab1:
        st.subheader("Matriz de confusión")
        st.caption("Permite identificar qué clases se están confundiendo más entre sí.")
        cm_fig = plotConfusionMatrix(cm_data)
        st.pyplot(
            cm_fig,
            width='stretch'
            )

    with tab2:
        st.subheader("Importancia de variables")
        st.caption("Factores que más pesan en la toma de decisiones del algoritmo.")
        fp_fig = plotFeatureImportances(fp_data)
        st.pyplot(
            fp_fig,
            width='stretch'
            )

elif seccion == "Predicción individual":
    st.title("Predicción de deserción estudiantil")
    st.caption("Ingrese los indicadores del estudiante para evaluar la probabilidad de deserción en el siguiente periodo.")

    # 1. El Formulario con mejor layout
    with st.form("formulario_prediccion"):
        st.subheader("Rendimiento semestral del estudiante", text_alignment='center')

        col_a, col_b = st.columns(2)
        with col_a:
            prom_periodo = st.slider(
                "Promedio del Periodo (0-10)",
                0.0, 10.0, 5.5)
            asistencia = st.slider(
                "Promedio de asistencia (0-100)",
                0.0, 100.0, 50.0)
            total_mat = st.number_input(
                "Total materias matriculadas",
                1, 10, 1)

        with col_b:
            total_reprobadas = st.number_input(
                "Total materias reprobadas",
                0, 10, 0)
            nivel = st.number_input(
                "Nivel / Semestre",
                1, 8, 1)
            repitencias = st.number_input(
                "Repitencias (**número de veces que ha repetido al menos una materia**)",
                0, 10, 0)

        submitted = st.form_submit_button(
            "Analizar Estudiante",
            use_container_width=True
            )

    if submitted:
        input_data = pd.DataFrame({
            'PROM_PERIODO': [prom_periodo],
            'ASIST_PROM': [asistencia],
            'TOTAL_MAT': [total_mat],
            'REPROBADAS': [total_reprobadas],
            'REPITENCIAS': [repitencias],
            'NIVEL': [nivel]
        })

        probabilidad, prediccion = dropoutPredictor.predict(input_data)

        st.divider()

        if prediccion == 1:
            st.markdown(f"""
                <div class="result-box high-risk">
                    <h2 style='margin:0;'>Riesgo de deserción</h2>
                    <p>El modelo detectó patrones críticos de deserción.</p>
                    <h1 style='margin:0;'>{probabilidad:.1%}</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box low-risk">
                    <h2 style='margin:0;'>Sin iesgo de deserción</h2>
                    <p>El estudiante presenta indicadores de estabilidad académica.</p>
                    <h1 style='margin:0;'>{probabilidad:.1%}</h1>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("¿Por qué el modelo predice esto?")

        with st.spinner("Analizando factores clave..."):
            fig_explain = dropoutPredictor.explain()

            st.pyplot(fig_explain, width='stretch')
