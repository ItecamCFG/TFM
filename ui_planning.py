# ui_planning.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta, date
import traceback  # Importa la biblioteca traceback
import time
import math


try:
    from model import solve_scheduling_problem, solve_scheduling_problem_scip
except ImportError:
    st.error("Error: No se pudo encontrar el archivo 'model.py'. Asegúrate de que exista.")
    # Función dummy si model.py falta (¡necesitará ser actualizada!)
    def solve_scheduling_problem(*args, **kwargs):
        st.warning("Función 'solve_scheduling_problem' no encontrada o no actualizada. Usando resultado dummy.")
        # Devolver datos dummy que *ignoran* secuencias y deadlines
        # ¡ESTO ES SOLO PARA VISUALIZACIÓN DE UI, NO REFLEJA LA LÓGICA CORRECTA!
        assignment = {
            ('Project Alpha', 'Analysis', 'Alice', 1): 8,
            ('Project Alpha', 'Analysis', 'Alice', 2): 8,
            ('Project Alpha', 'Analysis', 'Alice', 3): 4,
            ('Project Alpha', 'Development', 'Alice', 4): 8, # Ignora secuencia
            ('Project Beta', 'Design', 'Charlie', 1): 8,
        }
        return 0, assignment # Coste y asignación dummy

from data_manager import DAYS, EXPERTISE_LEVELS # Importar constantes

DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"] # Nombres de días para la disponibilidad


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
            options=["CBC (PuLP)", "SCIP (PySCIPOpt)"],
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

            # Paso 2: Ejecutar solver
            with st.spinner(f"🔍 Optimizando con {solver_choice.split(' ')[0]} (tiempo límite: 5 min)..."):
                start_time = time.time()

                if solver_choice == "CBC (PuLP)":
                    optimal_makespan, assignment, task_completion_times = solve_scheduling_problem(
                        projects_list=model_data['projects_original_list'],
                        tasks_list=model_data['tasks_original_list'],
                        resource_names_list=model_data['resources'],
                        expertise_dict=model_data['expertise_levels'],
                        cost_dict=model_data['costs'],
                        days_list=model_data['days'],
                        availability_numeric=model_data['availability'],
                        start_date=model_data['start_date']
                    )
                elif solver_choice == "SCIP (PySCIPOpt)":
                    # Esta parte la dejaremos para después, para mantener el foco en PuLP
                    optimal_cost, assignment = solve_scheduling_problem_scip(
                        model_data['projects'],
                        model_data['tasks_per_project'],
                        model_data['hours_required'],
                        model_data['expertise_required'],
                        model_data['resources'],
                        model_data['expertise_levels'],
                        model_data['costs'],
                        model_data['days'],
                        model_data['availability']
                    )

                # Guardar resultados en sesión
                st.session_state.update({
                    'last_run_success': True,
                    'optimal_makespan': optimal_makespan, # Cambio aquí
                    'assignment': assignment,
                    'task_completion_times': task_completion_times, # Cambio aquí
                    'model_metadata': {
                        'execution_time': round(time.time() - start_time, 2),
                        'solver_used': solver_choice,
                        'planning_horizon': model_data['planning_horizon']
                    }
                })

        except Exception as e:
            st.session_state['last_run_success'] = False
            st.error(f"❌ Error crítico durante la ejecución: {str(e)}")
            st.error(traceback.format_exc())  # Imprime el traceback completo

        # Mostrar resultados si existen
        show_results()

def prepare_solver_data(app_data):
    """Prepara y valida los datos para el modelo v5 (Makespan)."""
    # Inicializar diccionario de salida
    data = {
        'valid': False,
        # --- Datos que SÍ necesita la función v5 ---
        'projects_original_list': [],  # Lista de dicts de proyecto original
        'tasks_original_list': [],     # Lista de dicts de tarea original
        'resources': [],              # Lista de nombres de recursos
        'expertise_levels': {},        # Dict {recurso: expertis}
        'costs': {},                   # Dict {recurso: coste}
        'days': [],                    # Lista de números de día [1, ..., N]
        'availability': {},           # Dict {(recurso, día_num): horas}
        'start_date': None,            # Objeto date del día 1
        # --- Datos adicionales (pueden ser útiles para debug o metadata) ---
        'planning_horizon': 0,
        # --- Datos que NO necesita directamente la función v5 ---
        # 'projects': [],
        # 'tasks_per_project': {},
        # 'hours_required': {},
        # 'expertise_required': {},
    }
    today = date.today()  # <-- Obtener fecha de inicio
    data['start_date'] = today

    try:
        # =====================
        # 1. Proyectos
        # =====================
        projects = app_data.get('projects', [])
        if not projects:
            raise ValueError("No hay proyectos definidos")

        data['projects_original_list'] = projects  # <-- GUARDAR LISTA ORIGINAL
        project_names = [p['name'] for p in projects]
        project_id_to_name = {p['id']: p['name'] for p in projects}

        # =====================
        # 2. Tareas
        # =====================
        tasks = app_data.get('tasks', [])
        if not tasks:
            raise ValueError("No hay tareas definidas")

        data['tasks_original_list'] = tasks  # <-- GUARDAR LISTA ORIGINAL

        # Validar que todas las tareas pertenecen a proyectos conocidos
        for task in tasks:
            if task.get('project_id') not in project_id_to_name:
                st.warning(
                    f"Tarea '{task.get('name')}' tiene un ID de proyecto inválido: {task.get('project_id')}. Será ignorada.")
        # Filtrar tareas inválidas (opcional pero recomendado)
        tasks = [t for t in tasks if t.get('project_id') in project_id_to_name]
        if not tasks:
            raise ValueError("No quedan tareas válidas tras filtrar.")
        data['tasks_original_list'] = tasks  # Guardar lista filtrada

        # =====================
        # 3. Recursos
        # =====================
        resources = app_data.get('resources', [])
        if not resources:
            raise ValueError("No hay recursos definidos")

        resource_names = [r['name'] for r in resources]
        expertise_levels = {r['name']: r.get('expertise', "Junior") for r in resources}  # Default a Junior
        costs = {r['name']: r.get('cost', 0) for r in resources}
        availability_by_day_name = {
            r['name']: {day: r.get(day, 0) for day in DAYS}
            for r in resources
        }

        data['resources'] = resource_names
        data['expertise_levels'] = expertise_levels
        data['costs'] = costs

        # =====================
        # 4. Horizonte Temporal y Lista de Días
        # =====================
        # Calcular horizonte basado en deadlines (o usa tu lógica anterior si prefieres)
        latest_deadline = today
        if projects:
            deadlines = [p.get('deadline') for p in projects if isinstance(p.get('deadline'), date)]
            if deadlines:
                latest_deadline = max(deadlines)

        final_planning_date = latest_deadline + timedelta(days=5)
        if final_planning_date <= today:
            final_planning_date = today + timedelta(days=20)  # Mínimo 20 días

        planning_horizon = (final_planning_date - today).days + 1
        if planning_horizon <= 0:
            planning_horizon = 30

        days_list = list(range(1, planning_horizon + 1))

        data['days'] = days_list
        data['planning_horizon'] = planning_horizon  # Guardar para metadata

        # =====================
        # 5. Disponibilidad Numérica
        # =====================
        availability_numeric = {}
        start_weekday = today.weekday()  # 0=Lunes, 6=Domingo

        for r_name in resource_names:
            for d_num in days_list:
                current_weekday = (start_weekday + d_num - 1) % 7
                if 0 <= current_weekday < len(DAYS):  # Lunes-Viernes
                    day_name = DAYS[current_weekday]
                    hours = availability_by_day_name.get(r_name, {}).get(day_name, 0)
                    availability_numeric[(r_name, d_num)] = hours
                else:  # Sábado o Domingo
                    availability_numeric[(r_name, d_num)] = 0

        data['availability'] = availability_numeric
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

        if not df.empty:
            st.dataframe(
                df.sort_values(by=["Proyecto", "Día"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se encontraron asignaciones significativas")

    # Diagrama de Gantt
    if not df.empty:
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


# Por ahora esto se queda aquí, pero se puede mover a un archivo de configuración o similar
'''
def display_planning():
    """Muestra la UI para ejecutar la planificación y ver los resultados."""
    st.header("Planificación y Resultados")
    app_data = st.session_state.app_data

    if not app_data.get('tasks') or not app_data.get('resources'):
        st.warning("Añade proyectos con tareas y recursos antes de poder planificar.")
        return

    # --- Selector de Solver ---
    solver_choice = st.selectbox(
        "Selecciona el Solver a utilizar:",
        options=["CBC (PuLP)", "SCIP (PySCIPOpt)"],
        index=0
    )
    st.caption("SCIP suele ser más rápido para problemas complejos, pero requiere instalación. CBC es más lento pero suele venir incluido.")

    # Botón único con key fijo
    if st.button(
        f"🚀 Ejecutar Planificación con {solver_choice.split(' ')[0]}",
        key="unique_planning_button"
    ):
        # --- Preparar datos para el modelo ---
        with st.status("Preparando datos para el modelo...", expanded=True) as status:
            # Proyectos
            projects_list = app_data['projects']
            project_names_list = [p['name'] for p in projects_list]
            project_id_to_name = {p['id']: p['name'] for p in projects_list}
            project_name_to_id = {v: k for k, v in project_id_to_name.items()}

            # Tareas
            tasks_list = app_data['tasks']
            tasks_per_project_name = {p_name: [] for p_name in project_names_list}
            hours_required_dict = {}
            expertise_required_dict = {}
            for task in tasks_list:
                proj_id = task['project_id']
                proj_name = project_id_to_name.get(proj_id)
                if proj_name:
                    task_name = task['name']
                    if task_name not in tasks_per_project_name[proj_name]:
                        tasks_per_project_name[proj_name].append(task_name)
                        hours_required_dict[(proj_name, task_name)] = task['hours']
                        expertise_required_dict[(proj_name, task_name)] = task['expertise']

            # Recursos
            resources_list = app_data['resources']
            resource_names_list = [r['name'] for r in resources_list]
            expertise_dict = {r['name']: r['expertise'] for r in resources_list}
            cost_dict = {r['name']: r['cost'] for r in resources_list}
            availability_by_day_name = {
                r['name']: {day_name: r.get(day_name, 0) for day_name in DAY_NAMES}
                for r in resources_list
            }

            # Horizonte de planificación
            total_hours_required = sum(hours_required_dict.values()) or 0
            num_resources = len(resource_names_list)
            max_daily_capacity = num_resources * 8 if num_resources > 0 else 1
            estimated_min_days = total_hours_required / max_daily_capacity if max_daily_capacity > 0 else 1
            planning_horizon = int(estimated_min_days * 2) + 5
            days_list = list(range(1, planning_horizon + 1))

            # Disponibilidad numérica
            availability_numeric = {}
            start_weekday = 0
            for r_name in resource_names_list:
                for d_num in days_list:
                    current_weekday = (start_weekday + d_num - 1) % 7
                    if 0 <= current_weekday < len(DAY_NAMES):
                        day_name = DAY_NAMES[current_weekday]
                        availability_numeric[(r_name, d_num)] = availability_by_day_name.get(r_name, {}).get(day_name, 0)
                    else:
                        availability_numeric[(r_name, d_num)] = 0

            # Validaciones
            valid = True
            if not projects_list or not tasks_list or not resources_list:
                st.error("Faltan proyectos, tareas o recursos.")
                valid = False
            
            if not valid:
                status.update(label="Preparación cancelada", state="error", expanded=False)
                return

            status.update(label="Datos preparados correctamente!", state="complete", expanded=False)

        # --- Ejecutar solver ---
        status_container = st.empty()
        try:
            with status_container:
                st.warning("""
                **¡Importante!** La planificación necesita considerar:
                1. Secuencia de Tareas
                2. Deadlines de Proyecto
                3. Continuidad/Penalización Cambio
                """)
                
                if solver_choice == "CBC (PuLP)":
                    with st.spinner(f'Optimizando con CBC (tiempo límite: 5 min)...'):
                        optimal_cost, assignment = solve_scheduling_problem(
                            project_names_list,
                            tasks_per_project_name,
                            hours_required_dict,
                            expertise_required_dict,
                            resource_names_list,
                            expertise_dict,
                            cost_dict,
                            days_list,
                            availability_numeric
                        )
                elif solver_choice == "SCIP (PySCIPOpt)":
                    with st.spinner(f'Optimizando con SCIP (tiempo límite: 5 min)...'):
                        optimal_cost, assignment = solve_scheduling_problem_scip(
                            project_names_list,
                            tasks_per_project_name,
                            hours_required_dict,
                            expertise_required_dict,
                            resource_names_list,
                            expertise_dict,
                            cost_dict,
                            days_list,
                            availability_numeric
                        )

                # Guardar resultados
                st.session_state['last_run_success'] = True
                st.session_state['optimal_cost'] = optimal_cost
                st.session_state['assignment'] = assignment
                st.success("¡Planificación completada con éxito!")

        except Exception as e:
            st.session_state['last_run_success'] = False
            st.error(f"Error durante la ejecución: {str(e)}")
            st.error(traceback.format_exc())

    # Mostrar resultados
    if 'last_run_success' in st.session_state and st.session_state['last_run_success']:
        st.subheader("Resultados de la Planificación")
        optimal_cost = st.session_state.get('optimal_cost')
        assignment = st.session_state.get('assignment')

        if optimal_cost is not None:
            st.metric(label="**Coste Total Óptimo**", value=f"{optimal_cost:.2f} €")

        if assignment:
            # Generar tabla de asignación
            schedule_data = []
            for (proj, task, resource, day), hours in assignment.items():
                if hours > 0.01:
                    schedule_data.append({
                        "Proyecto": proj,
                        "Tarea": task,
                        "Recurso": resource,
                        "Día": day,
                        "Horas": round(hours, 2)
                    })
            
            if schedule_data:
                st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)
            else:
                st.info("No se encontraron asignaciones válidas")

    # --- Mostrar Resultados (si existen en el estado) ---
    if 'last_run_success' in st.session_state:
        st.subheader("Resultados de la Última Planificación")
        # (El código para mostrar resultados (tabla, Gantt) puede permanecer similar al anterior,
        # pero necesitará adaptarse al formato exacto que devuelva tu *nuevo* `assignment`.
        # El Gantt, en particular, debería ahora reflejar las secuencias si el modelo las calcula.)

        if st.session_state['last_run_success']:
            optimal_cost = st.session_state.get('optimal_cost')
            assignment = st.session_state.get('assignment') # Asume formato {(proj_name, task_name, resource_name, day): hours}

            if optimal_cost is not None and assignment is not None:
                if optimal_cost >= 0 and assignment : # Checkear coste y si hay asignaciones
                    st.metric(label="**Coste Total Óptimo Estimado**", value=f"{optimal_cost:.2f} €")

                    # --- Tabla de Asignación ---
                    st.subheader("Tabla de Asignación Detallada")
                    # (Mismo código que antes para crear schedule_df de assignment)
                    schedule_data = []
                    if isinstance(assignment, dict): # Verificar que assignment sea un diccionario
                        for key, hours in assignment.items():
                            # Asumiendo que la clave es una tupla (proyecto, tarea, recurso, día)
                            if isinstance(key, tuple) and len(key) == 4 and hours > 0.01:
                                proj, task_name, resource, day = key
                                schedule_data.append({
                                    "Proyecto": proj,
                                    "Tarea": task_name,
                                    "Recurso": resource,
                                    "Día": day,
                                    "Horas Asignadas": round(hours, 2)
                                })
                    if schedule_data:
                        schedule_df = pd.DataFrame(schedule_data)
                        st.dataframe(schedule_df, use_container_width=True)
                    else:
                        st.info("El modelo se ejecutó correctamente pero no se realizaron asignaciones significativas.")
                else:
                    st.info("No se encontraron asignaciones para mostrar.")
            else:
                st.warning("El formato de 'assignment' devuelto por el modelo no es el esperado (se esperaba un diccionario).")


            # --- Diagrama de Gantt (Simplificado - Necesitaría ajuste fino con modelo real) ---
            st.subheader("Diagrama de Gantt (Estimado)")
            if schedule_data:
                # (Mismo código que antes para crear gantt_df y figura px.timeline)
                # Crear el Gantt chart (código igual al de la versión anterior)
                print("Contenido de schedule_data:", schedule_data)  # Debug: Imprimir el contenido de schedule_data
                gantt_data = []
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                start_date -= timedelta(days=start_date.weekday())
                day_to_date = {i + 1: start_date + timedelta(days=i) for i in range(len(DAYS))}

                project_name_to_id = {p['name']: p['id'] for p in app_data.get('projects', [])}
                task_name_to_data = {(t['project_id'], t['name']): t for t in app_data.get('tasks', [])}

                task_resource_groups = {}

                for item in schedule_data:
                    proj_name = item['Proyecto']
                    task_name = item['Tarea']
                    resource = item['Recurso']
                    day = item['Día']
                    hours = item['Horas Asignadas']

                    proj_id = project_name_to_id.get(proj_name)
                    task_info = task_name_to_data.get((proj_id, task_name))
                    task_seq = task_info.get('sequence', 0) if task_info else 0

                    key = (proj_name, task_name, resource, task_seq) # Añadir secuencia a la clave
                    if key not in task_resource_groups:
                        task_resource_groups[key] = []
                    task_resource_groups[key].append({'day': day, 'hours': hours})


                for (proj, task, resource, seq), days_info in sorted(task_resource_groups.items(), key=lambda item: (item[0][0], item[0][3])): # Sort by project, then sequence
                    assigned_days = sorted([day_info['day'] for day_info in days_info])
                    if not assigned_days: continue

                    task_start_day_index = assigned_days[0] - 1
                    task_end_day_index = assigned_days[-1] - 1

                    # Ensure the index is within the bounds of DAYS
                    if 0 <= task_start_day_index < len(DAYS) and 0 <= task_end_day_index < len(DAYS):
                        task_start_day_name = DAYS[task_start_day_index]
                        task_end_day_name = DAYS[task_end_day_index]

                        # Find the actual date for the first and last day of the task
                        first_assignment_date = None
                        last_assignment_date = None
                        sorted_assignment_days = sorted([item['Día'] for item in schedule_data if item['Proyecto'] == proj and item['Tarea'] == task and item['Recurso'] == resource])

                        if sorted_assignment_days:
                            start_day_num = sorted_assignment_days[0]
                            end_day_num = sorted_assignment_days[-1]
                            first_assignment_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=start_day_num - 1)
                            last_assignment_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=end_day_num)


                        if first_assignment_date and last_assignment_date:
                            total_hours = sum(d['hours'] for d in days_info)
                            gantt_data.append(dict(
                                Task=f"[{seq}] {proj} - {task}", # Incluir secuencia en nombre
                                Start=first_assignment_date.strftime("%Y-%m-%d"),
                                Finish=last_assignment_date.strftime("%Y-%m-%d"),
                                Resource=resource,
                                Hours=round(total_hours,1)
                            ))

                if gantt_data:
                    gantt_df = pd.DataFrame(gantt_data)
                    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task",
                                        color="Resource", hover_data=["Resource", "Hours"],
                                        title="Planificación Estimada por Tarea y Recurso")
                    fig.update_yaxes(categoryorder='array', categoryarray=sorted(gantt_df['Task'].unique())) # Ordenar por secuencia/nombre
                    fig.update_layout(xaxis_title="Fecha", yaxis_title="[Secuencia] Proyecto - Tarea")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para generar el diagrama de Gantt.")


        else: # last_run_success == False
            st.error("La última ejecución de la planificación falló. Revisa los mensajes de error, los datos de entrada o el `model.py`.")

            '''