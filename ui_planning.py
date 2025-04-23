# ui_planning.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import date, timedelta, datetime
import traceback  # Importa la biblioteca traceback
import time

from optimization.data_models import OptimizationInput, Project, Task, Resource, OptimizationConfig, OptimizationResult
from optimization.model_pulp import MakespanMinimizationPuLP
from optimization.model_SCIP import MakespanMinimizationSCIP  # Placeholder para SCIP

# Mapeo de opciones del UI a clases de modelo
MODEL_MAPPING = {
    "Min (PuLP/CBC)": MakespanMinimizationPuLP,
    "Min (PySCIPOpt)":MakespanMinimizationSCIP,  # Placeholder para SCIP
    }

DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

def display_planning():
    """Función principal que maneja la UI de planificación"""
    st.header("📅 Planificación y Resultados")
    app_data = st.session_state.get('app_data', {})

    # Verificar datos mínimos requeridos
    if not app_data.get('projects') or not app_data.get('tasks') or not app_data.get('resources'):
        st.warning("⚠️ Añade proyectos, tareas y recursos antes de planificar")
        return

    # --- Selector de Solver ---
    col1, col2 = st.columns([3, 7])
    with col1:
        solver_choice = st.selectbox(
            "**Selecciona el motor de optimización:**",
            options=list(MODEL_MAPPING.keys()),
            index=0,
            key="solver_choice_selectbox"
        )
    with col2:
        st.caption("""
        - **CBC**: Solver por defecto de PuLP (más lento pero siempre disponible)
        - **SCIP**: Solver avanzado (requiere instalación manual pero más rápido)
        """)

    # --- Botón de ejecución ---
    run_button = st.button(
        f"🚀 Ejecutar Planificación con {solver_choice.split(' ')[0]}",
        key="unique_execution_button",
        type="primary"
    )

    if run_button:
        try:
            # Limpiar resultados anteriores
            if 'last_run_success' in st.session_state:
                del st.session_state['last_run_success']

            # Paso 1: Preparar datos
            with st.status("🔨 Preparando datos para el modelo...", expanded=True) as status:
                model_data = prepare_solver_data(app_data)
                if not model_data['valid']:
                    status.update(label="❌ Error en datos de entrada", state="error", expanded=False)
                    return
                status.update(label="✅ Datos preparados correctamente", state="complete", expanded=False)

            # Paso 2: Crear configuracion
            config = OptimizationConfig(
                start_date=model_data['start_date'],
                solver_time_limit=300,  # Límite de tiempo en segundos (5 min)
                planning_horizon_days=model_data['planning_horizon']  # Si se calcula aquí, descomentar
            )

            # Paso 3: Crear input del modelo
            input_data = OptimizationInput(
                projects=model_data['projects_original_list'],
                tasks=model_data['tasks_original_list'],
                resources=model_data['resources'],
                config=config
            )

            # Paso 4: Instanciar y resolver el modelo
            SelectedModelClass = MODEL_MAPPING[solver_choice]
            model_instance = SelectedModelClass(input_data)
            
            with st.spinner(f"🔍 Optimizando con {solver_choice.split(' ')[0]} (tiempo límite: 5 min)..."):
                result = model_instance.solve()  # Llamar al método solve del objeto
            
            # Paso 5: Guardar resultados
            st.session_state['last_result'] = result
            st.session_state['assignment'] = result.assignment
            st.session_state['optimal_makespan'] = result.makespan
            st.session_state['model_metadata'] = {
                'execution_time': round(result.solver_runtime, 2),
                'planning_horizon': model_data['planning_horizon']
            }
            st.session_state['last_run_success'] = True
            st.success("✅ Optimización finalizada. Mostrando resultados...")

        except Exception as e:
            st.error(f"❌ Error crítico durante la ejecución: {str(e)}")
            st.error(traceback.format_exc())  # Mostrar traceback completo
            st.session_state['last_result'] = OptimizationResult(status="Error", error_message=traceback.format_exc()) 
        
        # Mostrar resultados si existen
    show_results()

def prepare_solver_data(app_data):
    """Prepara y valida los datos para los modelos de optimización."""
    data = {
        'valid': False,
        'projects_original_list': [],
        'tasks_original_list': [],
        'resources': [],
        'start_date': None,
        'planning_horizon': 30,
    }

    today = date.today()
    data['start_date'] = today

    try:
        # ===================== 1. Proyectos =====================
        projects_raw = app_data.get('projects', [])
        if not projects_raw:
            raise ValueError("No hay proyectos definidos")

        projects = [Project(**p) for p in projects_raw]
        data['projects_original_list'] = projects
        project_ids = {p.id for p in projects}

        # ===================== 2. Tareas =====================
        tasks_raw = app_data.get('tasks', [])
        if not tasks_raw:
            raise ValueError("No hay tareas definidas")

        # Validar tareas con proyectos existentes
        tasks_valid = [t for t in tasks_raw if t.get('project_id') in project_ids]
        if not tasks_valid:
            raise ValueError("No quedan tareas válidas tras filtrar")

        tasks = [Task(**t) for t in tasks_valid]
        data['tasks_original_list'] = tasks

        # ===================== 3. Recursos =====================
        resources_raw = app_data.get('resources', [])
        if not resources_raw:
            raise ValueError("No hay recursos definidos")

        resources = []
        for r in resources_raw:
            availability = {day: r.get(day, 0) for day in DAYS}
            resources.append(Resource(
                name=r['name'],
                expertise=r.get('expertise', 'Junior'),
                cost=r.get('cost', 0.0),
                availability=availability
            ))
        data['resources'] = resources

        # ===================== 4. Horizonte de planificación =====================
        deadlines = [p.deadline for p in projects if isinstance(p.deadline, date)]
        latest_deadline = max(deadlines) if deadlines else today
        final_planning_date = max(latest_deadline + timedelta(days=5), today + timedelta(days=20))
        planning_horizon = (final_planning_date - today).days + 1
        data['planning_horizon'] = planning_horizon

        # ===================== 5. Estado final =====================
        data['valid'] = True

    except Exception as e:
        st.error(f"Error preparando datos: {str(e)}")
        data['valid'] = False

    return data


def show_results():
    """Muestra los resultados de la planificación (v5 - Makespan)."""
    if 'last_run_success' not in st.session_state:
        return

    st.divider()
    st.subheader("📊 Resultados de la Planificación")

    if not st.session_state['last_run_success']:
        st.error("La última ejecución falló. Verifica los logs de error.")
        return

    # Metadatos de ejecución
    metadata = st.session_state.get('model_metadata', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        optimal_makespan = st.session_state.get('optimal_makespan', None)
        if optimal_makespan is not None:
            st.metric("Makespan Óptimo", f"{optimal_makespan:.2f} días")  # Cambio aquí
        else:
            st.warning("No se pudo determinar el Makespan.")
    with col2:
        st.metric("Días Planificados", metadata.get('planning_horizon', 0))
    with col3:
        st.metric("Tiempo Ejecución", f"{metadata.get('execution_time', 0)}s")

    # Tabla de asignaciones
    assignment = st.session_state.get('assignment', {})
    df = None
    if assignment:
        st.subheader("🗓 Asignaciones Detalladas")
        df = pd.DataFrame([
            {
                "Proyecto": proj,
                "Tarea": task,
                "Recurso": resource,
                "Día": day,
                "Horas": hours
            }
            for (proj, task, resource, day), hours in assignment.items()
            if hours > 0.01  # Filtrar asignaciones mínimas
        ])

        if df is not None:
            st.dataframe(
                df.sort_values(by=["Proyecto", "Día"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se encontraron asignaciones significativas")

    # Diagrama de Gantt
    if df is not None:
        st.subheader("📈 Diagrama de Gantt")

        try:
            # Convertir días a fechas
            start_date = datetime.now().replace(hour=0, minute=0, second=0)
            df["Fecha Inicio"] = df["Día"].apply(
                lambda d: start_date + timedelta(days=d - 1)
            )
            df["Fecha Fin"] = df["Fecha Inicio"] + pd.to_timedelta(df["Horas"], unit='h')

            # Agrupar tareas continuas
            gantt_df = df.groupby(["Proyecto", "Tarea", "Recurso"]).agg({
                "Fecha Inicio": "min",
                "Fecha Fin": "max",
                "Horas": "sum"
            }).reset_index()

            # Crear visualización
            fig = px.timeline(
                gantt_df,
                x_start="Fecha Inicio",
                x_end="Fecha Fin",
                y="Proyecto",
                color="Recurso",
                hover_data=["Tarea", "Horas"],
                title="Planificación Temporal",
                labels={"Proyecto": "Proyecto", "Fecha Inicio": "Inicio", "Fecha Fin": "Fin"},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Proyecto",
                height=600,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"No se pudo generar el diagrama de Gantt: {str(e)}")
