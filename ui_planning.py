# ui_planning.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import date, timedelta, datetime
import traceback
import time

# --- Importar Clases ---
# Asegúrate que las rutas sean correctas según tu estructura
from optimization.data_models import OptimizationInput, OptimizationConfig, Project, Task, Resource, OptimizationResult
from optimization.base_model import OptimizationModel # Importar base si necesitas type hinting
from optimization.model_pulp import MakespanMinimizationPuLP
from optimization.neal_model import NealMakespanModel
from optimization.model_SCIP import MakespanMinimizationSCIP

# --- Mapeo de Opciones a Clases ---
MODEL_MAPPING = {
    "Minimizar Makespan (PuLP/CBC)": MakespanMinimizationPuLP,
    "Minimizar Makespan (Neal QUBO)": NealMakespanModel, 
    "Minimizar Makespan (PySCIPOpt)": MakespanMinimizationSCIP, 
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
                 status.update(label="✅ Datos listos", state="complete")

            # 3. Instanciar y Resolver
            SelectedModelClass = MODEL_MAPPING[model_choice]
            model_instance = SelectedModelClass(input_data)

            spinner_msg = f"🔍 Optimizando con {model_choice}..."
            if "Neal" in model_choice:
                 spinner_msg += " (Simulated Annealing)"
            else:
                 spinner_msg += f" (Límite: {time_limit}s)"

            with st.spinner(spinner_msg):
                result = model_instance.solve() # Llamar al método solve del objeto

            # 4. Guardar Resultado y la Fecha de Inicio USADA
            st.session_state['last_result'] = result # <-- GUARDAR EL OBJETO RESULTADO
            st.session_state['last_run_start_date'] = input_data.config.start_date # <-- GUARDAR LA FECHA DE INICIO USADA
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


# --- Función para Mostrar Resultados (Adaptada) ---
def show_results():
    """Muestra los resultados de la planificación leyendo de st.session_state."""
    if 'last_result' not in st.session_state:
        st.info("Ejecuta la planificación para ver los resultados.")
        return

    st.divider()
    st.subheader("📊 Resultados de la Planificación")

    result = st.session_state['last_result']

    # Mostrar Estado y Mensajes de Error/Advertencia
    if result.status == "Optimal":
        st.success(f"Solución Óptima encontrada en {result.solver_runtime:.2f}s.")
    elif result.status == "Timelimit":
        st.warning(f"Límite de tiempo alcanzado ({result.solver_runtime:.2f}s). Mostrando la mejor solución encontrada (puede no ser óptima).")
    elif result.status == "Feasible" or result.status == "Feasible (Validation Pending)":
         st.success(f"Solución factible encontrada por Neal en {result.solver_runtime:.2f}s.")
         if "Validation Pending" in result.status:
              st.warning("⚠️ **Importante:** La validez de esta solución QUBO no ha sido comprobada contra las restricciones originales.")
    elif "Error" in result.status:
         st.error(f"Falló la ejecución: {result.status}")
         if result.error_message:
              st.code(result.error_message)
         return # No mostrar más si hubo error grave
    elif result.status == "Infeasible":
         st.error("El modelo resultó ser infactible. Revisa los datos y restricciones.")
         return
    else: # Not Run, etc.
         st.info(f"Estado del solver: {result.status}")
         # return # Podrías salir aquí o intentar mostrar lo que haya

    # --- Métricas Principales ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if result.makespan is not None:
             st.metric("Makespan Estimado", f"{result.makespan:.0f} días")
        else:
             st.metric("Makespan Estimado", "N/A")
    with col2:
        # El coste total se debe calcular en _extract_results si se quiere mostrar
        if result.total_cost is not None:
             st.metric("Coste Total Estimado", f"€ {result.total_cost:.2f}")
        else:
             st.metric("Coste Total Estimado", "N/A")
    with col3:
         # Mostrar energía para Neal? O runtime?
         st.metric("Tiempo del Solver", f"{result.solver_runtime:.2f}s")
         # metadata = st.session_state.get('model_metadata', {}) # Ya no necesario si info está en result
         # st.metric("Días Planificados", metadata.get('planning_horizon', 'N/A'))


    # --- Tabla de Asignaciones ---
    assignment = result.assignment
    df = None
    if assignment:
        st.subheader("🗓 Asignaciones Detalladas")
        try:
            schedule_data = []
            for (proj, task, resource, day), hours in assignment.items():
                 if hours > 0.01: # Filtrar asignaciones mínimas
                      schedule_data.append({
                           "Proyecto": proj, "Tarea": task, "Recurso": resource,
                           "Día Num.": day, "Horas": round(hours, 2)
                       })
            if schedule_data:
                 df = pd.DataFrame(schedule_data)
                 st.dataframe(
                     df.sort_values(by=["Proyecto", "Día Num."]),
                     use_container_width=True,
                     hide_index=True
                 )
            else:
                 st.info("No se encontraron asignaciones significativas en la solución.")
        except Exception as e:
            st.error(f"Error al procesar asignaciones para la tabla: {e}")
            df = None # Asegurar que df es None si falla
    else:
        st.info("No hay datos de asignación disponibles.")

    # --- Diagrama de Gantt ---
    # --- Visualización Detallada por Recurso ---
    if df is not None and not df.empty:
        st.subheader("🛠 Visualización por Recurso")

        recurso_seleccionado = st.selectbox("Selecciona un recurso:", df["Recurso"].unique())
        gantt_start_date = st.session_state.get('last_run_start_date', date.today())
        df_recurso = df[df["Recurso"] == recurso_seleccionado].copy()
        df_recurso["Fecha"] = df_recurso["Día Num."].apply(lambda d: gantt_start_date + timedelta(days=d - 1))

        fig_barras = px.bar(
            df_recurso,
            x="Fecha",
            y="Horas",
            color="Proyecto",
            hover_data=["Tarea"],
            title=f"Horas asignadas por día para el recurso: {recurso_seleccionado}"
        )
        fig_barras.update_layout(xaxis_title="Fecha", yaxis_title="Horas asignadas")
        st.plotly_chart(fig_barras, use_container_width=True)
        
    # Mostrar asignaciones detalladas
    # Necesita start_date y task_completion_days para ser preciso,
    # o basarse solo en las asignaciones de 'assignment' (menos preciso para duración)
    if assignment: # Solo mostrar si hay asignaciones
        st.subheader("📈 Diagrama de Gantt (Estimado por Asignación)")
        try:
            gantt_data = []
            gantt_start_date = st.session_state.get('last_run_start_date', date.today())
            # Necesitamos la fecha de inicio usada por el modelo
            # start_date = st.session_state.get('last_run_start_date', date.today()) # Obtener de la nueva clave
            start_day = date.today() # Usar hoy como fecha de inicio por defecto
            # Agrupar por tarea y recurso para encontrar inicio/fin de bloques de trabajo
            if df is None: # Regenerar df si no se creó para la tabla por algún error previo
                schedule_data = []
                for (proj, task, resource, day), hours in assignment.items():
                    if hours > 0.01:
                        schedule_data.append({
                                "Proyecto": proj, "Tarea": task, "Recurso": resource,
                                "Día Num.": day, "Horas": round(hours, 2)
                            })
                if schedule_data:
                      df_gantt = pd.DataFrame(schedule_data)
                else:
                      st.info("No hay datos de asignación para generar el Gantt.")
                      return # Salir si no hay datos
            else:
                 df_gantt = df.copy() # Usar df de la tabla si existe

            df_gantt["Fecha"] = df_gantt["Día Num."].apply(lambda d: gantt_start_date + timedelta(days=d - 1))

            task_resource_groups = df_gantt.groupby(['Proyecto', 'Tarea', 'Recurso'])

            for name, group in task_resource_groups:
                proj, task, resource = name
                start_day_np = group['Día Num.'].min()
                end_day_np = group['Día Num.'].max()
                total_hours = group['Horas'].sum()

                start_day_int = int(start_day_np)
                end_day_int = int(end_day_np)

                task_start_date = gantt_start_date + timedelta(days=start_day_int - 1)
                task_end_date = gantt_start_date + timedelta(days=end_day_int - 1)

                gantt_data.append(dict(
                    Project = proj,
                    Task=f"{proj} - {task}", # Usar nombre combinado
                    Start=task_start_date.strftime("%Y-%m-%d"),
                    Finish=task_end_date.strftime("%Y-%m-%d"),
                    Resource=resource,
                    Hours=round(total_hours,1)
                 ))

            if gantt_data:
                 gantt_df_final = pd.DataFrame(gantt_data)
                 fig = px.timeline(gantt_df_final, x_start="Start", x_end="Finish", y="Task",
                                  color="Project",
                                  hover_data=["Resource", "Hours","Task"],
                                  title="Planificación Temporal Estimada",
                                  labels={"Task": "Proyecto - Tarea"}
                                  )
                 fig.update_yaxes(categoryorder='total ascending')
                 fig.update_layout(xaxis_title="Fecha", yaxis_title="Tarea")
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No se pudo generar datos para el diagrama de Gantt.")

        except Exception as e:
            st.warning(f"No se pudo generar el diagrama de Gantt: {str(e)}")
            st.error(traceback.format_exc()) # Descomentar para debug detallado