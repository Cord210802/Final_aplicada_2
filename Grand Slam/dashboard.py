import streamlit as st
from limpieza import main
import matplotlib.pyplot as plt
import seaborn as sns
import io
from contextlib import redirect_stdout
import os

# Configure Seaborn globally
sns.set(style="whitegrid", font_scale=1.2)

# --- Sidebar navigation ---
st.sidebar.header("Navegación")
page = st.sidebar.selectbox("Ir a:", ["Home", "Reporte", "Variables"])

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
    best_of_five = final_data.copy()

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