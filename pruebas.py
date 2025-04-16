# ui_planning.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import time
import traceback

# Importar Clases de Datos y Modelos
from data_models import OptimizationInput, OptimizationConfig, Project, Task, Resource
from optimization.makespan_model_pulp import MakespanMinimizationPuLP
# from optimization.cost_model_pulp import CostMinimizationPuLP # Si creas esta clase
# from optimization.makespan_model_scip import MakespanMinimizationSCIP # Si creas esta

# Mapeo de opciones del UI a clases de modelo
# ¡Asegúrate que las clases existen en las rutas importadas!
MODEL_MAPPING = {
    "Minimizar Makespan (PuLP/CBC)": MakespanMinimizationPuLP,
    # "Minimizar Coste (PuLP/CBC)": CostMinimizationPuLP, # Descomentar si existe
    # "Minimizar Makespan (PySCIPOpt)": MakespanMinimizationSCIP, # Descomentar si existe
}

def display_planning():
    st.header("Planificación y Resultados")
    app_data = st.session_state.app_data # Asume que app_data tiene 'projects', 'tasks', 'resources'

    # Validaciones básicas de datos de entrada
    if not app_data.get('tasks') or not app_data.get('resources') or not app_data.get('projects'):
        st.warning("⚠ Añade proyectos con tareas y recursos antes de poder planificar.")
        return

    # Selección del Modelo/Objetivo
    model_choice = st.selectbox(
        "Selecciona el Modelo/Objetivo a Optimizar:",
        options=list(MODEL_MAPPING.keys())
    )

    # Parámetros de Configuración (Ejemplo)
    # Podrías poner sliders o inputs aquí
    solver_time_limit = st.number_input("Límite de tiempo del Solver (segundos):", min_value=30, max_value=600, value=300, step=30)
    start_date_option = st.date_input("Fecha de inicio de la planificación:", value=date.today())
    # Calcular horizonte aquí o dentro del modelo base es una opción
    # planning_horizon_days = st.number_input("Horizonte de planificación (días):", min_value=30, value=90, step=15)

    if st.button(f"🚀 Ejecutar Planificación: {model_choice}"):
        # Limpiar resultados anteriores
        st.session_state.pop('last_result', None)

        try:
            # 1. Crear Objetos de Datos (si app_data usa dicts)
            #    Si ya usas objetos en app_data, este paso es más simple
            st.info("Preparando datos de entrada...")
            projects_obj = [Project(**p) for p in app_data.get('projects', [])]
            tasks_obj = [Task(**t) for t in app_data.get('tasks', [])]
            resources_obj = [Resource(**r) for r in app_data.get('resources', [])]
            # Asegurar que coste/disponibilidad en resources_obj sea correcto si es necesario

            # 2. Crear Configuración
            config = OptimizationConfig(
                start_date=start_date_option,
                solver_time_limit=solver_time_limit,
                # planning_horizon_days=planning_horizon_days # Pasar si se calcula aquí
            )

            # 3. Crear Input del Modelo
            input_data = OptimizationInput(
                projects=projects_obj,
                tasks=tasks_obj,
                resources=resources_obj,
                config=config
            )

            # 4. Instanciar y Resolver
            SelectedModelClass = MODEL_MAPPING[model_choice]
            model_instance = SelectedModelClass(input_data)

            with st.spinner(f"Optimizando con {model_choice}... (Límite: {solver_time_limit}s)"):
                result = model_instance.solve() # Llamar al método solve del objeto

            # 5. Guardar Resultado
            st.session_state['last_result'] = result
            st.success("Optimización finalizada.")

        except Exception as e:
            st.error(f"❌ Error crítico durante la ejecución: {str(e)}")
            st.error(traceback.format_exc())
            st.session_state['last_result'] = OptimizationResult(status="Error", error_message=traceback.format_exc())

        # Refrescar para mostrar resultados
        st.rerun()

    # --- Mostrar Resultados ---
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        st.subheader(f"Resultados ({result.status})")

        if result.status != "Error" and result.status != "Not Run":
             st.metric(label="**Tiempo del Solver (s)**", value=f"{result.solver_runtime:.2f}")
             if result.makespan is not None:
                 st.metric(label="**Makespan Óptimo Estimado (Días)**", value=f"{result.makespan:.0f}")
             if result.total_cost is not None: # Calcular coste total si es relevante
                 st.metric(label="**Coste Total Estimado (€)**", value=f"{result.total_cost:.2f}") # Necesitaría calcularse en _extract_results

             # Mostrar tabla de asignación (usando result.assignment)
             if result.assignment:
                  st.subheader("Tabla de Asignación Detallada")
                  schedule_df = pd.DataFrame(
                      [{'Proyecto': k[0], 'Tarea': k[1], 'Recurso': k[2], 'Día': k[3], 'Horas': round(v, 2)}
                       for k, v in result.assignment.items()]
                  )
                  st.dataframe(schedule_df, use_container_width=True)
             else:
                  st.info("No se encontraron asignaciones en la solución.")

             # Mostrar Gantt (necesitaría adaptar la lógica para usar result.assignment y/o result.task_completion_days)
             # ... (Código del Gantt adaptado) ...

        elif result.error_message:
             st.error("Falló la ejecución. Detalles del error:")
             st.code(result.error_message)