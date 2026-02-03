import matplotlib.pyplot as plt
import seaborn as sns


def plotConfusionMatrix(cm):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : pandas.DataFrame
        Confusion matrix where rows represent true labels and
        columns represent predicted labels.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the confusion matrix plot.
    """
    cm = cm.values

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
        )

    ax.set_xlabel("Predicción", fontsize=8)
    ax.set_ylabel("Valor Real", fontsize=8)
    ax.tick_params(labelsize=8)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig


def plotFeatureImportances(fp):
    """
    Plot feature importances using a horizontal bar chart.

    Parameters
    ----------
    fp : pandas.DataFrame
        DataFrame containing feature importance values.
        Expected columns are 'Feature' and 'Importance'.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the feature importance plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y='Feature',
        hue="Feature",
        data=fp,
        palette="viridis",
        ax=ax,
        legend=False
    )

    ax.set_xlabel("Puntaje de Importancia")
    ax.set_ylabel("")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig


def plotPromAverage(df):
    """
    Plot the distribution of the academic average per period.

    A histogram with a kernel density estimation (KDE) is used
    to visualize the distribution of student averages.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column 'PROM_PERIODO'.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the histogram plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        df['PROM_PERIODO'],
        kde=True,
        color="#4e79a7",
        ax=ax
        )
    ax.set_ylabel("")
    ax.set_xlabel("Promedio")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig


def plotAttendanceVsMean(df):
    """
    Plot the relationship between attendance and academic average.

    A scatter plot is generated to show how average attendance
    relates to academic performance, colored by dropout risk.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns 'ASIST_PROM',
        'PROM_PERIODO' and 'RIESGO_t1'.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the scatter plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    sns.scatterplot(
        data=df,
        x="ASIST_PROM",
        y="PROM_PERIODO",
        hue="RIESGO_t1",
        palette={0: "#2ecc71", 1: "#e74c3c"},
        alpha=0.6,
        s=40,
        ax=ax
    )

    ax.set_xlabel("Asistencia promedio (%)", fontsize=9)
    ax.set_ylabel("Promedio académico", fontsize=9)

    ax.tick_params(labelsize=8)

    ax.legend(
        title='Deserción (0: No, 1: Si)',
        fontsize='8',
        title_fontsize='9',
        loc='upper left',
        )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig


def plotStudentsLevel(df):
    """
    Plot the distribution of students across academic levels.

    A count plot is used to visualize the number of students
    per academic level.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column 'NIVEL'.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the count plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="NIVEL",
        hue="NIVEL",
        palette="viridis",
        ax=ax
        )
    ax.set_xlabel("Nivel")
    ax.set_ylabel("")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig
