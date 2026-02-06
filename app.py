import streamlit as st
import pandas as pd
import json
from src.models.dropoutPredictor import DropoutPredictor
from src.utils.figures import (
    plotConfusionMatrix, plotFeatureImportances,
    plotAttendanceVsMean, plotPromAverage,
    plotStudentsLevel
)
from src.utils.supabaseLogger import logPredictionSupabase


st.set_page_config(
    page_title="Predicción de deserción estudiantil",
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

            .rf-medium {
                background: linear-gradient(135deg, #fff8e1, #fffdf5);
                border-left: 6px solid #f9a825;
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
            .rf-medium .rf-title { color: #f57f17; }

            .rf-high .rf-title { color: #c62828; }

            .rf-value {
                font-size: 2.4rem;
                font-weight: 900;
            }

            .rf-low .rf-value { color: #1b5e20; }
            .rf-medium .rf-value { color: #e65100; }
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
        dp = DropoutPredictor(
            [
                "artifacts/dropout_rf_v2.joblib",
                "artifacts/dropout_lr_v2.joblib"
                ]
                )
        return dp
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None


dropoutPredictor = load_model()
if dropoutPredictor is None:
    st.stop()


@st.cache_data
def load_metrics(file_path):
    with open(file_path) as f:
        return json.load(f)


metrics = load_metrics("artifacts/metrics.json")


@st.cache_data
def get_cm_data(file_path):
    return pd.read_csv(file_path)


cm_data = get_cm_data("artifacts/confusion_matrix.csv")


@st.cache_data
def get_fp_data(file_path):
    return pd.read_csv(file_path)


fp_data = get_fp_data("artifacts/feature_importance.csv")


@st.cache_data
def load_data():
    df = pd.read_excel("data/processed/all_dataset.xlsx")
    return df


st.sidebar.markdown("## Navegación")

with st.sidebar:
    seccion = st.radio(
        label="Selecciona una opción",
        options=[
            "Análisis exploratorio",
            "Evaluación del modelo",
            "Predicción individual"
        ],
        index=0
    )

    st.divider()

    st.caption("By Dass")

if seccion == "Análisis exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")

    df = load_data()

    total_est = df['ESTUDIANTE'].nunique()
    tasa_riesgo = df['RIESGO_t1'].mean()
    prom_gral = df['PROM_PERIODO'].mean()

    with st.container(border=True):
        cols = st.columns(3)
        cols[0].metric(
            "Total estudiantes",
            f"{total_est}"
            )
        cols[1].metric(
            "Tasa de deserción",
            f"{tasa_riesgo:.2%}"
            )
        cols[2].metric(
            "Promedio general",
            f"{prom_gral:.2f}"
            )

    tab1, tab2, tab3 = st.tabs([
        "Rendimiento académico",
        "Distribución estudiantil",
        "Asistencia"
    ])

    with tab1:
        st.subheader("Distribución de promedios")
        st.caption("Muestra cómo se distribuyen los promedios académicos de los estudiantes.")
        st.pyplot(plotPromAverage(df), width="stretch")

    with tab2:
        st.subheader("Distribución de estudiantes por nivel")
        st.caption("Cantidad de estudiantes según su nivel académico.")
        st.pyplot(plotStudentsLevel(df), width="stretch")

    with tab3:
        st.subheader("Asistencia vs rendimiento")
        st.caption("Relación entre la asistencia a clases y el promedio académico.")
        st.pyplot(plotAttendanceVsMean(df), width="stretch")
        st.info(
            "Una menor asistencia está asociada a promedios más bajos "
            "y a una mayor presencia de estudiantes en riesgo de deserción."
        )

    with st.expander("Ver muestra de los datos"):
        st.dataframe(df.head(8), width="stretch")

elif seccion == "Evaluación del modelo":
    st.title("Evaluación del modelo")
    st.caption("Random Forest Classifier")

    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
            help="Porcentaje total de predicciones correctas."
            )
        cols[1].metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.2%}",
            help="Qué tan bien el modelo distingue entre estudiantes que desertan y los que no."
            )
        cols[2].metric(
            "Balanced Accuracy",
            f"{metrics['balanced_accuracy']:.2%}",
            help="Precisión considerando que hay más estudiantes de un tipo que de otro."
            )
        cols[3].metric(
            "F1-Score",
            f"{metrics['f1_score']:.2%}",
            help="Balance entre detectar bien a quienes desertan y evitar falsas alarmas."
            )

    tab1, tab2 = st.tabs([
        "Diagnóstico de errores",
        "Interpretación del modelo"
        ])

    with tab1:
        st.subheader("¿Dónde se equivoca el modelo?")
        st.caption("Este análisis muestra qué tan bien el modelo identifica a los estudiantes que podrían desertar.")

        st.markdown("### Matriz de confusión")
        st.caption("Compara las predicciones del modelo con la realidad.")

        cm_fig = plotConfusionMatrix(cm_data)
        st.pyplot(cm_fig, width="stretch")
        with st.expander("Ver métricas en detalle"):
            cols = st.columns(2)
            cols[0].metric(
                "Precision",
                f"{metrics['precission']:.2%}",
                help="Cuando el modelo predice deserción, ¿con qué frecuencia acierta?"
                )
            cols[1].metric(
                "Recall",
                f"{metrics['recall']:.2%}",
                help="De todos los estudiantes que realmente desertaron, ¿a cuántos logró detectar?"
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
        "basándose en patrones históricos."
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

        # try:
        #     logPredictionSupabase(
        #         dropoutPredictor.data,
        #         prob_rf,
        #         prob_lr,
        #         pred,
        #         dropoutPredictor.rf_threshold,
        #         dropoutPredictor.rf_version
        #     )
        # except Exception as e:
        #      st.warning("No se pudo registrar la predicción.")

        st.divider()
        LOW_RISK_LIMIT = 0.40
        THRESHOLD = dropoutPredictor.rf_threshold

        if prob_rf < LOW_RISK_LIMIT:
            risk_level = "low"
        elif prob_rf < THRESHOLD:
            risk_level = "medium"
        else:
            risk_level = "high"

        if risk_level == "low":
            risk_class = "rf-low"
            message = "No se detectan patrones significativos de riesgo académico."
        elif risk_level == "medium":
            risk_class = "rf-medium"
            message = (
                "Se identifican señales tempranas de riesgo académico. "
                "Se recomienda seguimiento preventivo."
            )
        else:
            risk_class = "rf-high"
            message = (
                "Se identifican indicadores fuertes asociados a riesgo de deserción académica."
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
