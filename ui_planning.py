# ui_planning.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import date, timedelta, datetime
import traceback
import time
from optimization.model_diagnostics import generate_model_diagnostics, generate_input_summary
from visualization.aux_visualizations import * # Importar funciones de visualización


# --- Importar Clases ---
# Asegúrate que las rutas sean correctas según tu estructura
from optimization.data_models import OptimizationInput, OptimizationConfig, Project, Task, Resource, OptimizationResult
from optimization.base_model import OptimizationModel # Importar base si necesitas type hinting
from optimization.model_pulp import MakespanMinimizationPuLP
from optimization.neal_model import NealMakespanModel
from optimization.model_SCIP import MakespanMinimizationSCIP
from optimization.genetic_model import GeneticMakespanPlanner
from utils import save_solver_result # Importar función para guardar resultados

# --- Mapeo de Opciones a Clases ---
MODEL_MAPPING = {
    "Minimizar Makespan (PuLP/CBC)": MakespanMinimizationPuLP,
    "Minimizar Makespan (Neal QUBO)": NealMakespanModel,
    "Minimizar Makespan (PySCIPOpt)": MakespanMinimizationSCIP,
    "Minimizar Makespan (Genético)": GeneticMakespanPlanner,
}

# --- Constantes ---
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"] # Solo días laborables para disponibilidad


# --- Función de Preparación (Simplificada/Opcional) ---
# Ahora la preparación principal está en _prepare_common_data de la clase base.
# Esta función solo necesita crear los objetos de datos si no lo haces ya en otros módulos UI.
def get_optimization_input(app_data, config: OptimizationConfig) -> OptimizationInput:
    """Crea el objeto OptimizationInput a partir de app_data."""
    # Validación básica
    if not app_data.get('projects') or not app_data.get('tasks') or not app_data.get('resources'):
        raise ValueError("Faltan datos esenciales (proyectos, tareas o recursos)")

    # Convertir dicts a objetos DataClass (si es necesario)
    try:
        st.info("Preparando datos para optimización...")
        projects_obj = [Project(**p) for p in app_data.get('projects', [])]
        # Convertir deadline de string a date si viene de JSON
        for p in projects_obj:
             if isinstance(p.deadline, str): p.deadline = date.fromisoformat(p.deadline)

        tasks_obj = [Task(**t) for t in app_data.get('tasks', [])]
        resources_raw = app_data.get('resources', [])
        resources_obj = []
        for r_dict in resources_raw:
             # Construimos el diccionario de disponibilidad solo con los dias esperados
             availability_dict = {day_name: r_dict.get(day_name,0) for day_name in DAYS}
             resource_name = r_dict.get('name')
             if not resource_name: # Si es None o está vacio
                  raise ValueError(f"Se encontro un recurso sin nombre en los datos:{r_dict}")
        # Crear el objeto Resource
             resources_obj.append(Resource(
                 name=r_dict.get('name', 'Desconocido'),
                 expertise=r_dict.get('expertise', 'Junior'),
                 cost=r_dict.get('cost', 0.0),
                 availability=availability_dict
             ))

    except Exception as e:
        raise ValueError(f"Error al convertir datos de app_data a objetos: {e}")

    return OptimizationInput(
        projects=projects_obj,
        tasks=tasks_obj,
        resources=resources_obj,
        config=config
    )


# --- Función Principal de la Pestaña ---
def display_planning():
    """Función principal que maneja la UI de planificación"""
    st.header("📅 Planificación y Resultados")
    # Obtener datos de la sesión (asume que tiene 'projects', 'tasks', 'resources')
    app_data = st.session_state.get('app_data', {})
    # st.write("🔍 Debug - Contenido de app_data:", app_data)

    # Verificar datos mínimos requeridos
    if not app_data.get('projects') or not app_data.get('tasks') or not app_data.get('resources'):
        st.warning("⚠️ Añade proyectos, tareas y recursos antes de planificar")
        return

    # --- Opciones de Configuración ---
    col_solver, col_time = st.columns(2)
    with col_solver:
        model_choice = st.selectbox(
            "**Modelo/Objetivo a Optimizar:**",
            options=list(MODEL_MAPPING.keys()),
            index=0,
            key="model_choice_selectbox"
        )
    with col_time:
        # El límite de tiempo aplica a solvers MILP, Neal tiene 'num_reads'
        # Podemos dejarlo aquí como referencia o hacerlo específico
        time_limit = st.number_input("Límite Tiempo MILP (s) / Ref. Neal:", min_value=30, max_value=1200, value=300, step=30)
        # num_reads = st.number_input("Número de Lecturas (Neal):", min_value=100, max_value=10000, value=1000, step=100)

    start_date_option = st.date_input("Fecha de inicio de la planificación:", value=date.today())

    # --- Botón de Ejecución ---
    if st.button(f"🚀 Ejecutar Planificación: {model_choice}", type="primary"):
        # Limpiar resultado anterior y fecha de inicio anterior
        st.session_state.pop('last_result', None)
        st.session_state.pop('last_run_start_date', None) # Limpiar la fecha de inicio anterior

        try:
            # 1. Crear Configuración
            config = OptimizationConfig(
                start_date=start_date_option,
                solver_time_limit=time_limit,
                # Añadir num_reads si es configurable:
                # num_reads=st.session_state.get('neal_num_reads', 100)
            )

            # 2. Crear Input del Modelo (validando dentro)
            with st.status("⚙️ Preparando y validando datos...", expanded=False) as status:
                 input_data = get_optimization_input(app_data, config)
                 st.session_state['input_data'] = input_data # Guardar el input_data en la sesión

                 status.update(label="✅ Datos listos", state="complete")

            # 3. Instanciar y Resolver
            SelectedModelClass = MODEL_MAPPING[model_choice]
            model_instance = SelectedModelClass(input_data)

            if 'input_data' in st.session_state:
                # Generar resumen de entrada
                generate_input_summary(input_data)

            spinner_msg = f"🔍 Optimizando con {model_choice}..."
            if "Neal" in model_choice:
                 spinner_msg += " (Simulated Annealing)"
            else:
                 spinner_msg += f" (Límite: {time_limit}s)"

            with st.spinner(spinner_msg):
                # Aquí puedes usar un contexto de progreso si es necesario
                # st.markdown(f"`Tipo del solver`: `{type(model_instance.model)}`") # DEBUG
                # Llamar a la función de optimización

                result = model_instance.solve() # Llamar al método solve del objeto
                dataset_name = st.session_state.get('dataset_name', 'manual_dataset')
                save_solver_result(result, input_data, dataset_name=dataset_name) # Guardar resultados en CSV

            
            # 4. Guardar Resultado y la Fecha de Inicio USADA
            st.session_state['last_result'] = result # <-- GUARDAR EL OBJETO RESULTADO
            st.session_state['last_model'] = model_instance # Guardar la instancia del modelo
            st.session_state['last_run_start_date'] = input_data.config.start_date # <-- GUARDAR LA FECHA DE INICIO USADA
            st.session_state['availability_numeric'] = model_instance.availability_numeric # Guardar la disponibilidad numérica
            st.success("✅ Optimización finalizada.")

        except ValueError as ve: # Capturar errores de validación de datos
             st.error(f"❌ Error en los Datos: {str(ve)}")
             st.session_state['last_result'] = OptimizationResult(status="Data Error", error_message=str(ve))
             # No guardamos last_run_start_date si hay error de datos, se usará date.today()
        except Exception as e:
            st.error(f"❌ Error crítico durante la ejecución: {str(e)}")
            st.error(traceback.format_exc())
            st.session_state['last_result'] = OptimizationResult(status="Execution Error", error_message=traceback.format_exc())
            # No guardamos last_run_start_date si hay error de ejecución

        # Refrescar para mostrar resultados
        st.rerun()

    # --- Mostrar Resultados ---
    # Llamar a la función show_results al final
    show_results()


def show_results():
    """Muestra los resultados de la planificación leyendo de st.session_state."""
    if 'last_result' not in st.session_state:
        st.info("Ejecuta la planificación para ver los resultados.")
        return

    st.divider()
    st.header("📊 Resultados de la Planificación")

    result = st.session_state['last_result']

    if 'last_model' in st.session_state:
        try:
            st.subheader("🔍 Diagnóstico del modelo")
            # st.write("🧪 DEBUG - Tipo de modelo:", type(st.session_state['last_model']))
            generate_model_diagnostics(st.session_state['last_model'])
        except Exception as diag_error:
            st.warning(f"No se pudo generar el diagnóstico del modelo: {diag_error}")

    # Estado de ejecución
    if result.status == "Optimal":
        st.success(f"Solución Óptima encontrada en {result.solver_runtime:.2f}s.")
    elif result.status == "Timelimit":
        st.warning(f"Límite de tiempo alcanzado ({result.solver_runtime:.2f}s). Mostrando la mejor solución encontrada.")
    elif result.status.startswith("Feasible"):
        st.success(f"Solución factible encontrada en {result.solver_runtime:.2f}s.")
        if "Validation Pending" in result.status:
            st.warning("⚠️ Solución QUBO no validada contra todas las restricciones.")
    elif "Error" in result.status:
        st.error(f"Falló la ejecución: {result.status}")
        if result.error_message:
            st.code(result.error_message)
        return
    elif result.status == "Infeasible":
        st.error("El modelo resultó infactible. Revisa los datos y restricciones.")
        return
    else:
        st.info(f"Estado del solver: {result.status}")

    # Métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Makespan Estimado", f"{result.makespan:.0f} días" if result.makespan is not None else "N/A")
    with col2:
        pass #st.metric("Coste Total Estimado", f"€ {result.total_cost:.2f}" if result.total_cost is not None else "N/A")
    with col3:
        st.metric("Tiempo del Solver", f"{result.solver_runtime:.2f}s")

    # Datos de asignación
    assignment = result.assignment
    if not assignment:
        st.info("No hay datos de asignación disponibles.")
        return

    schedule_data = [
        {"Proyecto": p, "Tarea": t, "Recurso": r, "Día Num.": d, "Horas": round(h, 2)}
        for (p, t, r, d), h in assignment.items() if h > 0.01
    ]
    df = pd.DataFrame(schedule_data)

    # Mapear expertise
    resource_expertise_map = {
        r.get("name", "Desconocido"): r.get("expertise", "Desconocido")
        for r in st.session_state.get("app_data", {}).get("resources", [])
    }
    df["Expertise"] = df["Recurso"].map(resource_expertise_map).fillna("Desconocido")

    # Visualizaciones
    gantt_start_date = st.session_state.get('last_run_start_date', date.today())

    # 1. Visualización por recurso
    st.subheader("🛠 Visualización por Recurso")
    recurso_seleccionado = st.selectbox("Selecciona un recurso:", df["Recurso"].unique(), key="select_recurso_principal")
    plot_resource_bar(df, recurso_seleccionado, gantt_start_date)

    # 2. Heatmap de recursos
    st.subheader("📊 Diagrama de ocupación de recursos")
    plot_resource_day_heatmap(df,key_prefix="Ocupacion")

    # 3. Diagrama de ocupación por recurso
    st.subheader("📋 Métricas de ocupación por recurso")
    availability_numeric = st.session_state.get('availability_numeric', {})
    show_idle_capacity(df, availability_numeric, makespan=result.makespan)




    # 4. Gantt por proyecto
    st.subheader("🧱 Visualización por Proyecto")
    proyecto_seleccionado = st.selectbox("Selecciona un proyecto:", df["Proyecto"].unique(), key="select_proyecto_gantt")
    plot_project_gantt(df, proyecto_seleccionado, gantt_start_date)

    # 5. Resumen por proyecto con gráfico de barras
    st.subheader("📊 Carga total por proyecto y recurso")
    plot_summary_by_project(df, resource_expertise_map, key_prefix="summary")

    # 6. Gantt global estimado
    st.subheader("📈 Diagrama de Gantt (Estimado por Asignación)")
    plot_assignment_gantt(df, gantt_start_date)

    return df.groupby("Proyecto").agg({"Horas": "sum"}).reset_index()

if __name__ == "__main__":
    # Solo para pruebas locales, no se ejecuta en Streamlit
    display_planning() # Descomentar para pruebas locales