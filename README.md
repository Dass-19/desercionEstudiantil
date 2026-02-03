# 📊 **Sistema de predicción de deserción estudiantil**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](#)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)](#)
[![CRISP-DM](https://img.shields.io/badge/Metodolog%C3%ADa-CRISP--DM-orange)](#)
[![Supabase](https://img.shields.io/badge/Supabase-Monitoring-3ECF8E?logo=supabase&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Deployed-success)](#)


# 📌 **Descripción del problema**
La deserción estudiantil es uno de los principales desafíos en la educación superior. 
La identificación temprana de estudiantes en riesgo permite implementar estrategias de 
intervención oportunas que favorezcan la permanencia académica.

Este proyecto presenta un sistema de análisis y predicción de deserción estudiantil, 
desarrollado mediante técnicas de minería de datos y aprendizaje automático, con una 
aplicación web interactiva orientada a la exploración, evaluación e interpretación de resultados.

# 🎯 **Descripción del proyecto**

A partir de un conjunto de datos académicos anonimizado, que incluye información como 
calificaciones, asistencia y trayectoria estudiantil, se construye un modelo de clasificación capaz de estimar el riesgo de deserción de los estudiantes.

El sistema permite:
- Analizar patrones académicos mediante análisis exploratorio de datos (EDA).
- Evaluar la relación entre asistencia, rendimiento académico y deserción.
- Medir el desempeño del modelo con métricas de clasificación.
- Interpretar los resultados de forma clara para usuarios no técnicos.

# 🚀 **Aplicación desplegada**

La aplicación se encuentra desplegada y operativa, permitiendo la interacción directa con los resultados del modelo sin necesidad de configuración local adicional.

🔗 Accede a la aplicación aquí: https://desercionestudiantil-ug.streamlit.app/

# 📂 **Estructura del Proyecto**

```text
.
├── 📂 data/                        # Datasets y archivos de datos crudos
├── 📂 artifacts/                   # Modelos y resultados
├── 📂 src/                         # Código fuente de la aplicación
│   ├── 📂 models/                  # Carga y lógica del modelo
│   ├── 📂 utils/                   # Funciones auxiliares y reutilizables           
│   ├── 📓 01_EDA.ipynb             # Exploración de los datos
│   ├── 📓 02_preprocessing.ipynb   # Preparación de los datos
│   └── 📓 03_modeling.ipynb        # Modelado
├── 🐍 app.py                       # Aplicación principal (Streamlit)
├── 🚫 .gitignore                   # Archivos excluidos de Git
├── 📖 README.md                    # Documentación del proyecto
└── 📋 requirements.txt             # Librerías y dependencias
```

# 🛠️ **Tecnologías utilizadas**

- **Lenguaje**: Python
- **Framework de aplicació**n: Streamlit
- **Análisis y procesamiento de datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Interpretabilidad y visualización**: Matplotlib, Seaborn, SHAP
- **Gestión de modelos**: Joblib
- **Integración de datos**: OpenPyXL
- **Monitoreo en producció**n: Supabase