import streamlit as st
import pandas as pd
import traceback
from datetime import date
from collections import defaultdict
import plotly.express as px

# Importamos las clases y constantes necesarias
from optimization.data_models import OptimizationInput, OptimizationConfig, Project, Task, Resource, OptimizationResult
from optimization.neal_model import NealMakespanModel
from optimization.dwave_hybrid_model import DWaveHybridMakespanModel  # Asegúrate de que este import sea correcto para tu estructura
# Asegúrate de que este import sea correcto para tu estructura
# from optimization.dwave_hybrid_model import DWaveHybridMakespanModel 
from data_manager import DAYS

# --- Mapeo de Modelos QUBO ---
# Añade aquí el modelo híbrido cuando lo tengas listo
QUBO_MODEL_MAPPING = {
    "Simulador Neal": NealMakespanModel,
    "D-Wave Híbrido": DWaveHybridMakespanModel, 
}

# --- Funciones de Parseo y Visualización ---

def parse_qubo_solution(result: OptimizationResult) -> dict:
    """
    Toma el objeto OptimizationResult de un modelo QUBO y lo traduce
    a un formato legible para la visualización.
    """
    if not result or not result.assignment:
        return {"assignments": [], "hours_worked": [], "makespan": "N/A", "completion_days": []}

    assignments_list = []
    seen_assignments = set()
    for (p, t, r, d), h in result.assignment.items():
        assignment_tuple = (p, t, r)
        if assignment_tuple not in seen_assignments:
            assignments_list.append({"Proyecto": p, "Tarea": t, "Recurso": r})
            seen_assignments.add(assignment_tuple)
    
    hours_list = [{"Día": d, "Proyecto": p, "Tarea": t, "Recurso": r, "Horas": round(h, 2)}
                  for (p, t, r, d), h in result.assignment.items()]
    
    completion_days_list = [{"Proyecto": p, "Tarea": t, "Día Finalización": int(day)}
                            for (p, t), day in result.task_completion_days.items()]

    return {
        "assignments": sorted(assignments_list, key=lambda x: (x['Proyecto'], x['Tarea'])),
        "hours_worked": sorted(hours_list, key=lambda x: (x['Día'], x['Proyecto'])),
        "makespan": result.makespan,
        "completion_days": sorted(completion_days_list, key=lambda x: (x['Proyecto'], x['Día Finalización']))
    }

def display_qubo_results():
    """Función auxiliar para mostrar los resultados del último experimento QUBO."""
    st.divider()
    st.subheader("📊 Resultados del Experimento")

    result = st.session_state.get('last_qubo_result')
    if not result:
        st.info("No hay resultados para mostrar. Ejecuta un experimento primero.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Estado Final", result.status)
    col2.metric("Energía / Objetivo", f"{result.objective_value:.2f}" if result.objective_value is not None else "N/A")
    col3.metric("Makespan (Días)", f"{result.makespan:.0f}" if result.makespan is not None else "N/A")

    if "Feasible" in result.status or "Optimal" in result.status:
        st.success("¡Se encontró una solución factible!")
    elif result.error_message:
        st.error("Ocurrió un error o la solución viola restricciones:")
        st.text(result.error_message)

    parsed_data = parse_qubo_solution(result)
    
    st.subheader("Mejor Solución Encontrada")

    if not parsed_data["assignments"]:
        st.warning("No se encontraron asignaciones en la mejor solución.")
        return

    # --- NUEVA VISUALIZACIÓN MEJORADA ---
    
    # 1. Pestañas para organizar la información
    tab_summary, tab_details = st.tabs(["Resumen de Planificación", "Detalle de Horas por Día"])

    with tab_summary:
        st.markdown("#### Resumen General")
        col_assign, col_days = st.columns(2)
        with col_assign:
            st.markdown("**Asignaciones de Tareas**")
            st.dataframe(pd.DataFrame(parsed_data['assignments']), hide_index=True)
        with col_days:
            st.markdown("**Días de Finalización**")
            st.dataframe(pd.DataFrame(parsed_data['completion_days']), hide_index=True)
        
        # Gráfico de Carga Total por Recurso
        st.markdown("---")
        st.markdown("**Carga de Trabajo Total por Recurso**")
        if parsed_data['hours_worked']:
            df_hours = pd.DataFrame(parsed_data['hours_worked'])
            df_load = df_hours.groupby("Recurso")["Horas"].sum().reset_index()
            
            fig = px.bar(df_load, x="Recurso", y="Horas", color="Recurso",
                         title="Horas Totales Asignadas por Recurso",
                         labels={'Horas': 'Horas Totales Acumuladas'})
            st.plotly_chart(fig, use_container_width=True)

    with tab_details:
        st.markdown("#### Desglose de Horas Trabajadas")
        if parsed_data['hours_worked']:
            df_hours = pd.DataFrame(parsed_data['hours_worked'])
            st.dataframe(df_hours, hide_index=True, use_container_width=True)
        else:
            st.info("No se registraron horas.")

# --- Interfaz Principal del Laboratorio ---
def display_qubo_lab():
    """
    Interfaz principal para el laboratorio de experimentos QUBO.
    VERSIÓN CON PREPARACIÓN DE DATOS CORREGIDA.
    """
    app_data = st.session_state.get('app_data', {})
    if not app_data.get('tasks'):
        st.warning("⚠️ Carga datos con proyectos, tareas y recursos para poder experimentar.")
        return

    st.subheader("Configuración de la Ejecución")
    
    model_choice = st.selectbox(
        "**Solver a Utilizar:**", options=list(QUBO_MODEL_MAPPING.keys()),
        index=0, key="qubo_model_selector"
    )

    col_date, col_horizon, col_param = st.columns(3)
    with col_date:
        start_date = st.date_input("Fecha de Inicio", value=date.today(), key="qubo_start_date")
    with col_horizon:
        horizon = st.number_input("Horizonte (días)", min_value=5, max_value=90, value=20, step=1, key="qubo_horizon")
    with col_param:
        if model_choice == "Simulador Neal":
            time_limit_or_reads = st.number_input("Número de Lecturas (reads)", min_value=100, max_value=10000, value=1000, step=100, key="qubo_reads")
        else: # Para D-Wave Híbrido
            time_limit_or_reads = st.number_input("Límite de Tiempo (s)", min_value=5, max_value=120, value=30, step=5, key="qubo_time_limit")
    
    if st.button(f"🚀 Ejecutar Experimento con {model_choice}", type="primary"):
        st.session_state['last_qubo_result'] = None
        
        with st.spinner(f"Ejecutando con {model_choice}..."):
            try:
                # --- ¡AQUÍ ESTÁ LA LÓGICA DE PREPARACIÓN DE DATOS ROBUSTA! ---
                projects_obj = []
                for p_dict in app_data.get('projects', []):
                    # Maneja tanto fechas que ya son objetos date como las que son strings
                    deadline = p_dict.get('deadline')
                    if isinstance(deadline, str):
                        try:
                            deadline = date.fromisoformat(deadline)
                        except (ValueError, TypeError):
                            deadline = None
                    p_dict['deadline'] = deadline
                    projects_obj.append(Project(**p_dict))
                
                tasks_obj = [Task(**t) for t in app_data.get('tasks', [])]
                
                resources_obj = []
                for r_dict in app_data.get('resources', []):
                    availability_dict = {day: r_dict.get(day, 0) for day in DAYS}
                    resources_obj.append(Resource(name=r_dict.get('name'), expertise=r_dict.get('expertise'), cost=r_dict.get('cost',0), availability=availability_dict))
                # --- FIN DE LA CORRECCIÓN ---

                config_params = {"start_date": start_date, "planning_horizon_days": horizon}
                if model_choice == "Simulador Neal":
                    config_params["num_reads"] = time_limit_or_reads
                else:
                    config_params["solver_time_limit"] = time_limit_or_reads

                config = OptimizationConfig(**config_params)
                input_data = OptimizationInput(projects=projects_obj, tasks=tasks_obj, resources=resources_obj, config=config)
                
                SelectedModelClass = QUBO_MODEL_MAPPING[model_choice]
                model_instance = SelectedModelClass(input_data)
                result = model_instance.solve()
                
                st.session_state['last_qubo_result'] = result
                st.success("Ejecución del experimento finalizada.")

            except Exception as e:
                st.error(f"❌ Error crítico durante el experimento: {e}")
                st.code(traceback.format_exc())
                st.session_state['last_qubo_result'] = OptimizationResult(status="Execution Error", error_message=traceback.format_exc())
            
        st.rerun()

    if st.session_state.get('last_qubo_result'):
        display_qubo_results()