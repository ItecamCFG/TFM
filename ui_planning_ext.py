import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

def enhanced_visualizations(df, gantt_start_date):
    if df is None or df.empty:
        st.info("No hay datos suficientes para visualización extendida.")
        return

    # Transformar día numérico en fecha
    df["Fecha"] = df["Día Num."].apply(lambda d: gantt_start_date + timedelta(days=d - 1))

    st.subheader("🛠 Visualización por Recurso")

    recurso_seleccionado = st.selectbox("Selecciona un recurso:", df["Recurso"].unique())

    df_recurso = df[df["Recurso"] == recurso_seleccionado].copy()

    fig_barras = px.bar(
        df_recurso,
        x="Fecha",
        y="Horas",
        color="Proyecto",
        hover_data=["Tarea"],
        title=f"Horas asignadas por día - {recurso_seleccionado}"
    )
    fig_barras.update_layout(xaxis_title="Fecha", yaxis_title="Horas")
    st.plotly_chart(fig_barras, use_container_width=True)

    st.subheader("📈 Carga Total por Día (Todos los Recursos)")
    df_total = df.groupby("Fecha")["Horas"].sum().reset_index()
    fig_total = px.line(df_total, x="Fecha", y="Horas", markers=True, title="Carga total diaria")
    fig_total.update_layout(xaxis_title="Fecha", yaxis_title="Horas asignadas")
    st.plotly_chart(fig_total, use_container_width=True)

    st.subheader("🧭 Mapa de Carga Recurso vs Día (Heatmap)")
    pivot = df.pivot_table(index="Recurso", columns="Fecha", values="Horas", aggfunc="sum", fill_value=0)
    fig_heatmap = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                            labels=dict(x="Fecha", y="Recurso", color="Horas"),
                            title="Horas asignadas por recurso y día")
    st.plotly_chart(fig_heatmap, use_container_width=True)
