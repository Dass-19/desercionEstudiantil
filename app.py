import streamlit as st
import pandas as pd
import json
from src.models.dropoutPredictor import DropoutPredictor
from src.utils.figures import (
    plotConfusionMatrix,
    plotFeatureImportances,
    plotAttendanceVsMean,
    plotPromAverage,
    plotStudentsLevel
)
from src.utils.supabaseLogger import logPredictionSupabase


RF_MODEL_PATH = st.secrets["RF_MODEL_PATH"]
LR_MODEL_PATH = st.secrets["LR_MODEL_PATH"]
METRICS_PATH = st.secrets["METRICS_PATH"]
CM_PATH = st.secrets["CM_PATH"]
FP_PATH = st.secrets["FP_PATH"]
PROCESSED_DATA_PATH = st.secrets["ALL_PROCESSED_DATA_PATH"]
LOG_PATH = st.secrets["LOG_PATH"]


st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    layout="wide")

st.markdown("""
            <style>
            .rf-card {
                border-radius: 16px;
                padding: 1.6rem 2.2rem;
                box-shadow: 0 8px 22px rgba(0,0,0,0.10);
                margin: 1.2rem 0;
            }

            .rf-low {
                background: linear-gradient(135deg, #e8f5e9, #f6fff7);
                border-left: 6px solid #2e7d32;
            }

            .rf-high {
                background: linear-gradient(135deg, #ffebee, #fff5f5);
                border-left: 6px solid #c62828;
            }

            .rf-title {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }

            .rf-low .rf-title { color: #2e7d32; }
            .rf-high .rf-title { color: #c62828; }

            .rf-value {
                font-size: 2.4rem;
                font-weight: 900;
            }

            .rf-low .rf-value { color: #1b5e20; }
            .rf-high .rf-value { color: #8e0000; }

            .rf-text {
                margin-top: 0.6rem;
                font-size: 0.95rem;
                color: #333;
            }
            </style>
            """,
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        dp = DropoutPredictor([RF_MODEL_PATH, LR_MODEL_PATH])
        return dp
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None


dropoutPredictor = load_model()
if dropoutPredictor is None:
    st.stop()


@st.cache_data
def load_metrics():
    with open(str(METRICS_PATH)) as f:
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
    df = pd.read_excel(str(PROCESSED_DATA_PATH))
    return df


st.sidebar.markdown("## Navegación")

with st.sidebar:
    seccion = st.radio(
        label="",
        options=[
            "Análisis exploratorio",
            "Evaluación del modelo",
            "Predicción individual"
        ],
        index=2
    )

    st.divider()

    st.caption("Sistema de apoyo académico")

if seccion == "Análisis exploratorio":
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
        st.subheader("Distribución de promedios", text_alignment='center')
        st.pyplot(
            plotPromAverage(df),
            width='stretch'
            )

    with col_right:
        st.subheader("Distirbución de estudiantes por nivel", text_alignment='center')
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

elif seccion == "Evaluación del modelo":
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

    st.info(
        "Esta herramienta estima el riesgo de deserción **para el siguiente periodo académico**, "
        "basándose en patrones históricos. No reemplaza la evaluación institucional."
        )

    with st.form("formulario_prediccion"):
        st.subheader("Rendimiento semestral del estudiante", text_alignment='center')

        col_a, col_b = st.columns(2)
        with col_a:
            prom_periodo = st.slider(
                "Promedio del Periodo",
                0.0, 10.0, 5.5,
                help="Promedio final del último periodo académico (escala 0 a 10)"
            )
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
                "Repitencias",
                0, 10, 0,
                help="Número de veces que el estudiante ha repetido al menos una asignatura"
            )

        submitted = st.form_submit_button(
            "Predecir deserción",
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
        with st.spinner("Calculando riesgo de deserción..."):
            prob_rf, pred, prob_lr = dropoutPredictor.predict(input_data)

        try:
            logPredictionSupabase(
                prob_rf,
                prob_lr,
                pred,
                dropoutPredictor.rf_threshold,
                dropoutPredictor.rf_version
            )
        except Exception as e:
             st.warning("No se pudo registrar la predicción.")

        st.divider()
        risk_class = "rf-high" if prob_rf >= dropoutPredictor.rf_threshold else "rf-low"

        message = (
            "Se identifican indicadores asociados a riesgo de deserción académica."
            if prob_rf >= dropoutPredictor.rf_threshold
            else "No se detectan patrones significativos de riesgo académico."
        )

        st.subheader("Resultado de la evaluación")

        st.markdown(f"""
        <div class="rf-card {risk_class}">
            <div class="rf-title">Random Forest · Riesgo estimado</div>
            <div class="rf-value">{prob_rf*100:.1f}%</div>
            <div class="rf-text">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Factores que influyen en la predicción")
        st.caption("Variables con mayor impacto según el modelo entrenado.")
        with st.spinner("Analizando factores clave..."):
            fig = dropoutPredictor.explain()
            st.pyplot(fig, )
