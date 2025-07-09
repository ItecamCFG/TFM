import streamlit as st
import pandas as pd
import traceback
from datetime import date
from collections import defaultdict

# Importamos las clases necesarias
from optimization.data_models import OptimizationInput, OptimizationConfig, Project, Task, Resource, OptimizationResult
from optimization.neal_model import NealMakespanModel
from optimization.dwave_hybrid_model import DWaveHybridMakespanModel # <-- Importamos el nuevo modelo
from data_manager import DAYS

# --- Mapeo de Modelos QUBO ---
QUBO_MODEL_MAPPING = {
    "Simulador Neal": NealMakespanModel,
    "D-Wave Híbrido": DWaveHybridMakespanModel,
}

def parse_qubo_solution(result: OptimizationResult) -> dict:
    """
    Toma el objeto OptimizationResult de un modelo QUBO y lo traduce
    a un formato legible para la visualización.
    """
    if not result or not result.assignment:
        return {
            "assignments": [], "hours_worked": [], "makespan": "N/A", "completion_days": []
        }

    # Extraer asignaciones de la estructura de resultados
    assignments_list = []
    # Usamos un set para no duplicar asignaciones de tarea-recurso
    seen_assignments = set()

    for (p, t, r, d), h in result.assignment.items():
        assignment_tuple = (p, t, r)
        if assignment_tuple not in seen_assignments:
            assignments_list.append({"Proyecto": p, "Tarea": t, "Recurso": r})
            seen_assignments.add(assignment_tuple)
    
    # Extraer horas trabajadas
    hours_list = [
        {"Día": d, "Proyecto": p, "Tarea": t, "Recurso": r, "Horas": h}
        for (p, t, r, d), h in result.assignment.items()
    ]
    
    # Extraer días de finalización
    completion_days_list = [
        {"Proyecto": p, "Tarea": t, "Día Finalización": day}
        for (p, t), day in result.task_completion_days.items()
    ]

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

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Estado Final", result.status)
    col2.metric("Energía / Objetivo", f"{result.objective_value:.2f}" if result.objective_value is not None else "N/A")
    col3.metric("Makespan (Días)", f"{result.makespan:.0f}" if result.makespan is not None else "N/A")

    if "Feasible" in result.status or "Optimal" in result.status:
        st.success("¡Se encontró una solución factible!")
    elif result.error_message:
        st.error("Ocurrió un error durante la ejecución:")
        st.code(result.error_message)

    # Parsear la mejor solución encontrada
    parsed_data = parse_qubo_solution(result)

    # Mostrar la solución en un formato claro
    st.subheader("Mejor Solución Encontrada")

    if not parsed_data["assignments"]:
        st.warning("No se encontraron asignaciones en la mejor solución.")
        return

    col_assign, col_days, col_hours = st.columns([1, 1, 1.5])

    with col_assign:
        st.markdown("**Asignaciones de Tareas**")
        st.dataframe(pd.DataFrame(parsed_data['assignments']), hide_index=True)

    with col_days:
        st.markdown("**Días de Finalización**")
        st.dataframe(pd.DataFrame(parsed_data['completion_days']), hide_index=True)
    
    with col_hours:
        st.markdown("**Desglose de Horas Trabajadas**")
        if parsed_data['hours_worked']:
            df_hours = pd.DataFrame(parsed_data['hours_worked'])
            st.dataframe(df_hours, hide_index=True, height=300)
        else:
            st.info("No se registraron horas.")


def display_qubo_lab():
    """
    Interfaz principal para el laboratorio de experimentos QUBO,
    permitiendo seleccionar y ejecutar diferentes solvers cuánticos/clásicos.
    """
    
    app_data = st.session_state.get('app_data', {})
    if not app_data.get('tasks'):
        st.warning("⚠️ Carga datos con proyectos, tareas y recursos para poder experimentar.")
        return

    st.subheader("Configuración de la Ejecución")
    
    # Selector de modelo
    model_choice = st.selectbox(
        "**Solver a Utilizar:**",
        options=list(QUBO_MODEL_MAPPING.keys()),
        index=0,
        key="qubo_model_selector"
    )

    # Parámetros de configuración
    col_date, col_horizon, col_param = st.columns(3)
    with col_date:
        start_date = st.date_input("Fecha de Inicio", value=date.today(), key="qubo_start_date")
    with col_horizon:
        horizon = st.number_input("Horizonte (días)", min_value=5, max_value=90, value=30, step=1, key="qubo_horizon")
    with col_param:
        # El parámetro cambia según el solver
        if model_choice == "Simulador Neal":
            time_limit_or_reads = st.number_input("Número de Lecturas (reads)", min_value=100, max_value=10000, value=1000, step=100, key="qubo_reads")
        else: # Para D-Wave Híbrido
            time_limit_or_reads = st.number_input("Límite de Tiempo (s)", min_value=5, max_value=120, value=30, step=5, key="qubo_time_limit")
    
    if st.button(f"🚀 Ejecutar Experimento con {model_choice}", type="primary"):
        st.session_state['last_qubo_result'] = None
        
        with st.spinner(f"Ejecutando con {model_choice}... Esto puede tardar unos minutos."):
            try:
                # 1. Preparar datos de entrada (común para ambos)
                projects_obj = []
                for p_dict in app_data.get('projects', []):
                    if isinstance(p_dict.get('deadline'), str): p_dict['deadline'] = date.fromisoformat(p_dict['deadline'])
                    projects_obj.append(Project(**p_dict))
                
                tasks_obj = [Task(**t) for t in app_data.get('tasks', [])]
                
                resources_obj = []
                for r_dict in app_data.get('resources', []):
                    availability_dict = {day: r_dict.get(day, 0) for day in DAYS}
                    resources_obj.append(Resource(name=r_dict.get('name'), expertise=r_dict.get('expertise'), cost=r_dict.get('cost', 0.0), availability=availability_dict))

                # 2. Configurar y seleccionar el modelo
                SelectedModelClass = QUBO_MODEL_MAPPING[model_choice]
                
                # Pasamos el límite de tiempo a la configuración general
                config = OptimizationConfig(
                    start_date=start_date, 
                    planning_horizon_days=horizon,
                    solver_time_limit=time_limit_or_reads if model_choice == "D-Wave Híbrido" else 30 # Default time limit
                )

                input_data = OptimizationInput(projects=projects_obj, tasks=tasks_obj, resources=resources_obj, config=config)
                
                # 3. Instanciar y resolver
                model_instance = SelectedModelClass(input_data)
                
                # Nota: Neal no usa `solver_time_limit`, tiene su propio sistema.
                # El modelo de D-Wave sí lo usará.
                result = model_instance.solve()
                
                # 4. Guardar resultados
                st.session_state['last_qubo_result'] = result
                st.success("Ejecución del experimento finalizada.")

            except Exception as e:
                st.error(f"❌ Error crítico durante el experimento: {e}")
                st.code(traceback.format_exc())
                st.session_state['last_qubo_result'] = OptimizationResult(status="Execution Error", error_message=traceback.format_exc())
            
        st.rerun()

    if st.session_state.get('last_qubo_result'):
        display_qubo_results()