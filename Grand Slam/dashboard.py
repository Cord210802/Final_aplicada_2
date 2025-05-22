import streamlit as st
import streamlit.components.v1 as components
from limpieza import main
import matplotlib.pyplot as plt
import seaborn as sns
import io
from contextlib import redirect_stdout
import os
import pandas as pd

# Agregar imports para análisis lineal\ nimport numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from scipy import stats
from matplotlib.gridspec import GridSpec
import numpy as np
# Configure Seaborn globally
sns.set(style="whitegrid", font_scale=1.2)

# --- Sidebar navigation ---
st.sidebar.header("Navegación")
page = st.sidebar.selectbox("Ir a:", ["Home", "Reporte", "Variables", "Linear model"])

if page == "Home":
    # --- Home page content ---
    st.title("Grand Slam Tennis Match Analysis Dashboard")

    # Sidebar settings for Home
    st.sidebar.header("Dashboard Settings")
    file_path = "/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data/atp_matches_till_2022.csv"
    year_start, year_end = st.sidebar.slider(
        "Year Range",
        1992, 2021,
        (2000, 2020),
        1
    )

    # Capture stdout from main for later display
    stdout_buffer = io.StringIO()
    plt.close('all')
    with redirect_stdout(stdout_buffer):
        final_data = main(
            file_path,
            show_plots=True,
            year_start=year_start,
            year_end=year_end
        )
    pipeline_output = stdout_buffer.getvalue()

    # Top-level summary
    st.markdown(
        f"""
    ## EDA de nuestro Dataset

    **Rango de años:** {year_start} – {year_end}  
    **Total de partidos cargados:** **{len(final_data)}**

    ---
    """,
        unsafe_allow_html=False
    )

    # Prepare best_of_five copy
        
    # ruta de la carpeta donde quieres guardar el CSV
    dir_path = "/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data"

    # nombre del archivo CSV
    filename = "best_of_five.csv"

    # construye la ruta completa
    output_path = os.path.join(dir_path, filename)

    # escribe el DataFrame sin la columna de índice
    final_data.to_csv(output_path, index=False)
    
    best_of_five = pd.read_csv("/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data/best_of_five.csv")

    # Create three tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Plots",
        "🖥️ VIF analysis",
        "🔗 Correlations"
    ])

    # --- Tab 1: Plots ---
    with tab1:
        st.subheader("Pipeline Visualizations")
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            st.pyplot(fig)

        # KDE + Boxplot for 'minutes'
        fig_kde_box, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(14, 10),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        sns.kdeplot(
            data=best_of_five,
            x="minutes",
            fill=True,
            color="#1f77b4",
            linewidth=2,
            alpha=0.7,
            ax=axes[0]
        )
        axes[0].set_title("Distribución de minutos para Grand Slams", fontsize=18, pad=20)
        axes[0].set_xlabel("Duración del partido (minutos)", fontsize=14)
        axes[0].set_ylabel("Densidad", fontsize=14)

        sns.boxplot(
            data=best_of_five,
            x="minutes",
            color="#1f77b4",
            linewidth=1.5,
            ax=axes[1]
        )
        axes[1].set_xlabel("Duración del partido (minutos)", fontsize=14)
        axes[1].set_title("Boxplot de duración de partidos", fontsize=16)

        plt.tight_layout()
        st.pyplot(fig_kde_box)

    # --- Tab 2: VIF analysis ---
    with tab2:
        st.subheader("See the VIF analysis")
        with st.expander("VIF scores", expanded=False):
            st.code(pipeline_output)
        lines = pipeline_output.splitlines()
        summary = [line for line in lines if "Dropping high VIF" in line or "Final dataset shape" in line]
        if summary:
            st.markdown("**Insights:**")
            for s in summary:
                st.markdown(f"- `{s}`")

    # --- Tab 3: Correlations ---
    with tab3:
        st.subheader("Top correlations with 'minutes' (Best of 5)")
        numeric_cols_5 = best_of_five.select_dtypes(include="number")
        corr_matrix_5 = numeric_cols_5.corr()
        correlations_with_minutes_5 = (
            corr_matrix_5["minutes"]
            .sort_values(key=lambda x: x.abs(), ascending=False)
        )
        st.table(correlations_with_minutes_5)

elif page == "Reporte":
    # --- Reporte page content ---
    st.title("Reporte Completo")
    md_path = "/home/cord2108/ITAM/Aplicada/Proyecto_final/reporte.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            report_md = f.read()
        st.markdown(report_md, unsafe_allow_html=False)
    else:
        st.error(f"No se encontró el archivo de reporte en:\n{md_path}")

elif page == "Variables":
    # --- Variables page content ---
    st.title("Variables")
    md_path_vars = "/home/cord2108/ITAM/Aplicada/Proyecto_final/variables.md"
    if os.path.exists(md_path_vars):
        with open(md_path_vars, "r", encoding="utf-8") as f:
            vars_md = f.read()
        st.markdown(vars_md, unsafe_allow_html=False)
    else:
        st.error(f"No se encontró el archivo de variables en:\n{md_path_vars}")
        
elif page == "Linear model":
    st.title("Diagnósticos del Modelo Lineal")

    # Mostrar markdown de descripción
    md_path_lm = "/home/cord2108/ITAM/Aplicada/Proyecto_final/LM_S.md"
    if os.path.exists(md_path_lm):
        with open(md_path_lm, "r", encoding="utf-8") as f:
            report_md = f.read()
        st.markdown(report_md, unsafe_allow_html=False)
    else:
        st.error(f"No se encontró el archivo de modelo lineal en:\n{md_path_lm}")

    # Cargar datos y ajustar modelo OLS
    df_lm = pd.read_csv(
        "/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data/best_of_five.csv"
    )
    y = df_lm['minutes']
    X = df_lm.drop(columns=['minutes'])
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # Resumen y Durbin–Watson
    st.subheader("Resumen OLS")
    # Inyectar estilo para fondo blanco y texto negro
    styled_html = f"""
    <style>
        table {{ background-color: white !important; color: black !important; }}
        th, td {{ background-color: white !important; color: black !important; }}
        .dataframe_title {{ color: #17202A; }}
        .statsmodels {{ font-family: Arial, sans-serif; }}
    </style>
    {model.summary().as_html(include_plotting_data=False)}
    """
    components.html(styled_html, height=600, scrolling=True)
    dw = sms.durbin_watson(model.resid)
    st.write(f"Durbin–Watson: {dw:.3f}")

    # Cálculos diagnósticos
    influence = model.get_influence()
    fitted = model.fittedvalues
    resid = model.resid
    std_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks = np.array(OLSInfluence(model).cooks_distance[0])
    # Gráficos diagnósticos
    # (resto del código de gráficos sin cambios)
    # 1. Residuos vs Ajustados
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    low = sm.nonparametric.lowess(resid, fitted)
    ax1.scatter(fitted, resid, alpha=0.7, color='#1F618D', s=60, edgecolor='white')
    ax1.plot(low[:,0], low[:,1], '--', color='#154360', lw=2)
    ax1.axhline(0, linestyle=':', color='#7F8C8D', lw=1.5)
    ax1.set_xlabel("Valores ajustados", fontweight='bold')
    ax1.set_ylabel("Residuos", fontweight='bold')
    st.pyplot(fig1)

    # 2. Q–Q Plot
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    q_theoretical = np.linspace(0.001, 0.999, len(resid))
    y_sorted = np.sort(resid)
    x_theor = stats.norm.ppf(q_theoretical)
    ax2.scatter(x_theor, y_sorted, alpha=0.7, color='#1F618D', s=60, edgecolor='white')
    min_val = min(x_theor.min(), y_sorted.min())
    max_val = max(x_theor.max(), y_sorted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], '--', color='#154360', lw=2)
    ax2.set_xlabel("Cuantiles teóricos", fontweight='bold')
    ax2.set_ylabel("Cuantiles observados", fontweight='bold')
    st.pyplot(fig2)

    # 3. Escala–Local
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    low2 = sm.nonparametric.lowess(np.sqrt(np.abs(std_resid)), fitted)
    ax3.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.7, color='#1F618D', s=60, edgecolor='white')
    ax3.plot(low2[:,0], low2[:,1], '--', color='#154360', lw=2)
    ax3.set_xlabel("Valores ajustados", fontweight='bold')
    ax3.set_ylabel("√|Residuos estandarizados|", fontweight='bold')
    st.pyplot(fig3)

    # 4. Leverage vs Residuos estandarizados
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    low3 = sm.nonparametric.lowess(std_resid, leverage)
    ax4.scatter(leverage, std_resid, alpha=0.7, color='#1F618D', s=60, edgecolor='white')
    ax4.plot(low3[:,0], low3[:,1], '--', color='#154360', lw=2)
    ax4.axhline(0, linestyle=':', color='#7F8C8D', lw=1.5)
    thresh_lev = float(2 * (X.shape[1]+1) / len(X))
    ax4.axvline(x=thresh_lev, linestyle='--', color='#CB4335', linewidth=1)
    ax4.set_xlabel("Leverage", fontweight='bold')
    ax4.set_ylabel("Residuos estandarizados", fontweight='bold')
    st.pyplot(fig4)

    # 5. Distancia de Cook
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    umbral_cook = 4/len(cooks)
    ax5.bar(np.arange(len(cooks)), cooks, alpha=0.3, color='#5DADE2')
    markerline, stemlines, _ = ax5.stem(np.arange(len(cooks)), cooks, markerfmt=",")
    plt.setp(stemlines, color='#2471A3', linewidth=2)
    ax5.axhline(y=umbral_cook, linestyle='--', color='#CB4335', linewidth=1.5)
    ax5.set_xlabel("Índice de observación", fontweight='bold')
    ax5.set_ylabel("Distancia de Cook", fontweight='bold')
    st.pyplot(fig5)

    # 6. Histograma de residuos
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    bins = 25
    ax6.hist(resid, bins=bins, density=True, alpha=0.7, color='#5DADE2', edgecolor='white')
    x_vals = np.linspace(resid.min(), resid.max(), 1000)
    ax6.plot(x_vals, stats.norm.pdf(x_vals, loc=resid.mean(), scale=resid.std()), '--', color='#154360', lw=2)
    ax6.axvline(x=resid.mean(), linestyle='-', color='#154360', linewidth=1.5)
    ax6.set_xlabel("Residuos", fontweight='bold')
    ax6.set_ylabel("Densidad", fontweight='bold')
    st.pyplot(fig6)
