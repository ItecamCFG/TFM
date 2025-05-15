import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta, date
import uuid
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

def plot_resource_day_heatmap(df, title="📅 Carga de recursos por día", key_prefix="heatmap"):
    if df is None or df.empty:
        st.info("No hay datos disponibles para graficar.")
        return

    # Agrupar por día y recurso para obtener horas totales asignadas
    summary = df.groupby(["Recurso", "Día Num."])["Horas"].sum().reset_index()

    recursos = sorted(df["Recurso"].unique())
    dias = sorted(df["Día Num."].unique())

    # Construir matriz z (horas asignadas), y matriz de texto
    z = []
    hover_text = []
    for r in recursos:
        row = []
        row_text = []
        for d in dias:
            match = summary[(summary["Recurso"] == r) & (summary["Día Num."] == d)]
            horas = match["Horas"].values[0] if not match.empty else 0.0
            row.append(horas)
            row_text.append(f"{r}<br>Día {d}<br>Horas: {horas:.1f}")
        z.append(row)
        hover_text.append(row_text)

    # Crear heatmap con escala personalizada de verde a azul
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=dias,
        y=recursos,
        text=hover_text,
        hoverinfo="text",
        zmin=0,
        zmax=8,
        colorscale=[
            [0.0, "#D2F8D2"],  # Verde claro (disponible)
            [0.25, "#B0D7F8"],
            [0.5, "#61A5F8"],
            [0.75, "#2C73D2"],
            [1.0, "#003F88"]   # Azul oscuro (ocupado)
        ],
        colorbar=dict(title="Horas asignadas")
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Día de planificación", type="category"),
        yaxis=dict(title="Recurso", type="category"),
        height=300 + 20 * len(recursos)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_intensity")




def plot_assignment_table(df):
    st.subheader("🗓 Asignaciones Detalladas")
    if df is not None and not df.empty:
        st.dataframe(
            df.sort_values(by=["Proyecto", "Día Num."]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No se encontraron asignaciones significativas en la solución.")

def plot_resource_bar(df, recurso_seleccionado, start_date):
    df_recurso = df[df["Recurso"] == recurso_seleccionado].copy()
    df_recurso["Fecha"] = df_recurso["Día Num."].apply(lambda d: start_date + timedelta(days=d - 1))
    fig_barras = px.bar(
        df_recurso,
        x="Fecha",
        y="Horas",
        color="Proyecto",
        hover_data=["Tarea"],
        title=f"Horas asignadas por día para el recurso: {recurso_seleccionado}"
    )
    fig_barras.update_layout(xaxis_title="Fecha", yaxis_title="Horas asignadas")
    st.plotly_chart(fig_barras, use_container_width=True, key=f"{recurso_seleccionado}_bar_chart")

def plot_project_gantt(df, proyecto_seleccionado, start_date):
    df_proyecto = df[df["Proyecto"] == proyecto_seleccionado].copy()
    df_proyecto["Fecha"] = df_proyecto["Día Num."].apply(lambda d: start_date + timedelta(days=d - 1))
    gantt_data = []
    for tarea, grupo in df_proyecto.groupby("Tarea"):
        start = grupo["Fecha"].min()
        end = grupo["Fecha"].max()
        recurso = grupo["Recurso"].iloc[0]
        gantt_data.append({
            "Tarea": tarea,
            "Inicio": start,
            "Fin": end,
            "Recurso": recurso
        })
    df_gantt_proj = pd.DataFrame(gantt_data)
    fig_proj = px.timeline(
        df_gantt_proj, x_start="Inicio", x_end="Fin", y="Tarea", color="Recurso",
        title=f"Ejecutando tareas en paralelo - {proyecto_seleccionado}"
    )
    fig_proj.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_proj, use_container_width=True,key=f"{proyecto_seleccionado}_gantt_chart")

def plot_summary_by_project(df, resource_expertise_map=None, key_prefix="summary"):
    if resource_expertise_map is None:
        resource_expertise_map = {}

    recursos_unicos = df["Recurso"].unique().tolist()
    recurso_seleccionado = st.selectbox(
        "Filtrar por Recurso (opcional):",
        ["Todos"] + recursos_unicos,
        key=f"{key_prefix}_recurso_filter_selectbox"
    )

    expertise_seleccionado_general = st.selectbox(
        "Filtrar por Nivel de Expertise (opcional):",
        ["Todos"] + sorted(df["Expertise"].dropna().unique().tolist()),
        key=f"{key_prefix}_expertise_filter_general"
    )

    df_filtered = df.copy()
    if recurso_seleccionado != "Todos":
        df_filtered = df_filtered[df_filtered["Recurso"] == recurso_seleccionado]
    if expertise_seleccionado_general != "Todos":
        df_filtered = df_filtered[df_filtered["Expertise"] == expertise_seleccionado_general]

    df_summary = df_filtered.groupby("Proyecto").agg({"Horas": "sum"}).reset_index()
    df_summary["Horas"] = df_summary["Horas"].round(2)
    df_summary = df_summary.sort_values(by="Horas", ascending=False)

    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # --- Gráfico por recurso
    st.markdown("### Detalle por Recurso dentro de cada Proyecto")
    df_detailed = df_filtered.groupby(["Proyecto", "Recurso"]).agg({"Horas": "sum"}).reset_index()
    df_detailed["Horas"] = df_detailed["Horas"].round(2)

    fig_detailed = px.bar(
        df_detailed,
        x="Proyecto",
        y="Horas",
        color="Recurso",
        title="Distribución de Horas por Proyecto y Recurso",
        labels={"Horas": "Horas Asignadas"},
        hover_data=["Recurso"]
    )
    fig_detailed.update_layout(barmode="stack")
    st.plotly_chart(fig_detailed, use_container_width=True,key=f"{key_prefix}_bar_detailed")

    # --- Gráfico agrupado por expertise
    if resource_expertise_map:
        df_detailed["Expertise"] = df_detailed["Recurso"].map(resource_expertise_map)
        expertise_seleccionado_exp = st.selectbox(
            "🔍 Filtrar por Nivel de Expertise (solo en vista agrupada):",
            ["Todos"] + sorted(df_detailed["Expertise"].dropna().unique().tolist()),
            key=f"{key_prefix}_expertise_filter_exp"
        )
        if expertise_seleccionado_exp != "Todos":
            df_detailed = df_detailed[df_detailed["Expertise"] == expertise_seleccionado_exp]

        st.markdown("### Vista Agrupada por Expertise")
        fig_exp = px.bar(
            df_detailed,
            x="Proyecto",
            y="Horas",
            color="Expertise",
            title="Distribución de Horas por Expertise y Proyecto",
            labels={"Horas": "Horas Asignadas"},
            hover_data=["Recurso"]
        )
        fig_exp.update_layout(barmode="stack")
        st.plotly_chart(fig_exp, use_container_width=True,key=f"{key_prefix}_bar_expertise")


def plot_assignment_gantt(df: pd.DataFrame, start_date):
    """Genera un diagrama de Gantt global combinando proyecto, tarea y recurso."""
    if df.empty:
        st.info("No hay datos para el Gantt de asignaciones.")
        return

    df["Fecha"] = df["Día Num."].apply(lambda d: start_date + timedelta(days=d - 1))

    gantt_data = []
    for (proj, tarea, recurso), grupo in df.groupby(["Proyecto", "Tarea", "Recurso"]):
        gantt_data.append({
            "Task": f"{proj} - {tarea}",
            "Start": grupo["Fecha"].min(),
            "Finish": grupo["Fecha"].max(),
            "Project": proj,
            "Resource": recurso,
            "Horas": grupo["Horas"].sum()
        })

    df_gantt = pd.DataFrame(gantt_data)

    if df_gantt.empty:
        st.info("No se pudo construir el diagrama de Gantt.")
        return

    fig = px.timeline(
        df_gantt,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Project",
        hover_data=["Resource", "Horas"],
        title="Planificación Temporal Estimada"
    )
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Tarea")
    st.plotly_chart(fig, use_container_width=True,key="assignment_gantt_chart")




def show_idle_capacity(df_asignaciones: pd.DataFrame, availability_numeric: dict, makespan: float):
    """Muestra análisis de capacidad ociosa por recurso y gráfico de ocupación."""

    if df_asignaciones.empty or not availability_numeric:
        st.warning("No hay datos suficientes para calcular la capacidad ociosa.")
        return

    recursos = df_asignaciones["Recurso"].unique()
    resultados = []

    for recurso in recursos:
        horas_asignadas = df_asignaciones[df_asignaciones["Recurso"] == recurso]["Horas"].sum()

        # Fix: convertir makespan a entero
        horas_disponibles = sum(
            availability_numeric.get((recurso, dia), 0) for dia in range(1, int(makespan) + 1)
        )

        horas_libres = horas_disponibles - horas_asignadas
        porcentaje_uso = (horas_asignadas / horas_disponibles) * 100 if horas_disponibles > 0 else 0.0

        resultados.append({
            "Recurso": recurso,
            "Horas Disponibles": round(horas_disponibles, 2),
            "Horas Asignadas": round(horas_asignadas, 2),
            "Horas Libres": round(horas_libres, 2),
            "Porcentaje Uso (%)": round(porcentaje_uso, 2)
        })

    df_result = pd.DataFrame(resultados)
    st.dataframe(df_result, use_container_width=True, hide_index=True)

    # Gráfico de barras del porcentaje de uso
    st.markdown("### 📊 Nivel de Ocupación por Recurso (en %)")
    fig = px.bar(
        df_result.sort_values("Porcentaje Uso (%)"),
        x="Porcentaje Uso (%)",
        y="Recurso",
        orientation="h",
        color="Porcentaje Uso (%)",
        color_continuous_scale="blues",
        labels={"Porcentaje Uso (%)": "Nivel de Ocupación (%)"},
        title="Porcentaje de Ocupación de cada Recurso"
    )
    fig.update_layout(xaxis_title="Porcentaje de Ocupación", yaxis_title="Recurso")
    st.plotly_chart(fig, use_container_width=True)



