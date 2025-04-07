# ui_planning.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta, date
import traceback  # Importa la biblioteca traceback

# Asume que model.py está en el mismo directorio o en PYTHONPATH
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
        return 2500.00, assignment # Coste y asignación dummy

from data_manager import DAYS, EXPERTISE_LEVELS # Importar constantes
DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"] # Nombres de días para la disponibilidad

def display_planning():
    """Muestra la UI para ejecutar la planificación y ver los resultados."""
    st.header("Planificación y Resultados")
    app_data = st.session_state.app_data

    if not app_data.get('tasks') or not app_data.get('resources'):
        st.warning("Añade proyectos con tareas y recursos antes de poder planificar.")
        return

    # --- NUEVO: Selector de Solver ---
    solver_choice = st.selectbox(
        "Selecciona el Solver a utilizar:",
        options=["CBC (PuLP)", "SCIP (PySCIPOpt)"],
        index=0 # Por defecto CBC
    )
    st.caption("SCIP suele ser más rápido para problemas complejos, pero requiere instalación. CBC es más lento pero suele venir incluido.")


    if st.button(f"🚀 Ejecutar Planificación con {solver_choice.split(' ')[0]}"):
        # --- Preparar datos para el modelo (v2.1 - CON CORRECCIÓN DE AVAILABILITY) ---
        st.info("Preparando datos para el modelo...")

        # Proyectos: Lista de diccionarios (sin cambios aquí)
        projects_list = app_data['projects'] # [{'id': ..., 'name':..., 'deadline':...}]
        project_names_list = [p['name'] for p in projects_list] # Nombres para PuLP/SCIP si los usan así
        project_id_to_name = {p['id']: p['name'] for p in projects_list}
        project_name_to_id = {v: k for k, v in project_id_to_name.items()} # Inverso

        # Tareas: Lista de diccionarios (sin cambios aquí)
        tasks_list = app_data['tasks'] # [{'id': ..., 'project_id':..., 'name':..., 'hours':..., 'expertise':..., 'sequence':...}]

        # Crear estructuras necesarias por el modelo (adaptado de tu PuLP original)
        tasks_per_project_name = {p_name: [] for p_name in project_names_list}
        hours_required_dict = {}
        expertise_required_dict = {}
        for task in tasks_list:
            proj_id = task['project_id']
            proj_name = project_id_to_name.get(proj_id)
            if proj_name:
                task_name = task['name']
                 # Evitar duplicados por nombre si la estructura lo requiere
                if task_name not in tasks_per_project_name[proj_name]:
                     tasks_per_project_name[proj_name].append(task_name)
                     hours_required_dict[(proj_name, task_name)] = task['hours']
                     expertise_required_dict[(proj_name, task_name)] = task['expertise']


        # Recursos: Lista de diccionarios (sin cambios aquí)
        resources_list = app_data['resources'] # [{'name':..., 'expertise':..., 'cost':..., 'Lunes':..., ...}]
        resource_names_list = [r['name'] for r in resources_list]
        expertise_dict = {r['name']: r['expertise'] for r in resources_list}
        cost_dict = {r['name']: r['cost'] for r in resources_list}
        # Guardamos la disponibilidad por nombre de día para el mapeo
        availability_by_day_name = {
            r['name']: {day_name: r.get(day_name, 0) for day_name in DAY_NAMES}
            for r in resources_list
        }

        # --- Calcular Horizonte de Planificación (days) y Disponibilidad Numérica ---
        # Calcular N_days: desde hoy hasta la última deadline + buffer (ej. 7 días)
        today = date.today()
        latest_deadline = today
        if app_data.get('projects'):
            deadlines = [p.get('deadline') for p in app_data['projects'] if p.get('deadline')]
            if deadlines:
                latest_deadline = max(deadlines)

        # Añadir un buffer (ej. 1 mes) por si acaso o si no hay deadlines
        final_planning_date = latest_deadline + timedelta(days=30)
        if final_planning_date <= today: # Asegurar al menos unos días si deadline es pasado/hoy
             final_planning_date = today + timedelta(days=30)

        n_days = (final_planning_date - today).days + 1 # Incluir hoy
        if n_days <= 0: n_days = 30 # Mínimo 30 días si algo falla

        days_list = list(range(1, n_days + 1)) # Lista de números de día [1, 2, ..., N]

        # Crear availability_numeric {(resource, day_number): hours}
        availability_numeric = {}
        start_weekday = today.weekday() # 0=Lunes, 6=Domingo

        for r_name in resource_names_list:
            for d_num in days_list:
                # Calcular qué día de la semana es d_num (0=Lunes, ..., 6=Domingo)
                current_weekday = (start_weekday + d_num - 1) % 7
                if 0 <= current_weekday < len(DAY_NAMES): # Es Lunes-Viernes?
                    day_name = DAY_NAMES[current_weekday]
                    # Obtener horas de la estructura original
                    hours = availability_by_day_name.get(r_name, {}).get(day_name, 0)
                    availability_numeric[(r_name, d_num)] = hours
                else: # Es Sábado o Domingo
                    availability_numeric[(r_name, d_num)] = 0 # Disponibilidad 0

        # --- Validaciones Previas (igual que antes) ---
        # ... (tu código de validación) ...
        valid = True
        if not projects_list or not tasks_list or not resources_list:
             st.error("Faltan proyectos, tareas o recursos.")
             valid = False
        # ... (más validaciones si las tienes) ...
        if not valid:
             st.warning("Corrige los errores en los datos antes de planificar.")
             return


        # --- Llamada al Solver Seleccionado ---
        st.warning("""
        **¡Importante!** La planificación aún necesita considerar:
        1.  **Secuencia de Tareas:** No implementado en `model.py`.
        2.  **Deadlines de Proyecto:** No implementado en `model.py`.
        3.  **Continuidad/Penalización Cambio:** No implementado en `model.py`.

        **Asegúrate de que tu `model.py` se actualice para estas restricciones.**
        """)
        st.info(f"Ejecutando el modelo con {solver_choice} (Límite de tiempo: 300s)...")
        optimal_cost = None
        assignment = {}

        with st.spinner(f'Buscando la mejor planificación con {solver_choice}... (puede tardar)'):
            try:
                if solver_choice == "CBC (PuLP)":
                    # Pasar los datos en el formato que espera la función PuLP original
                     optimal_cost, assignment = solve_scheduling_problem(
                         project_names_list,
                         tasks_per_project_name,
                         hours_required_dict,
                         expertise_required_dict,
                         resource_names_list,
                         expertise_dict,
                         cost_dict,
                         days_list, # Pasar lista de números de día
                         availability_numeric # Pasar disponibilidad numérica
                     )
                elif solver_choice == "SCIP (PySCIPOpt)":
                     # Pasar los datos en el formato que espera la función PySCIPOpt
                     # (Asegúrate de que coincida, puede que necesites ajustar ligeramente)
                     optimal_cost, assignment = solve_scheduling_problem_scip(
                         project_names_list,
                         tasks_per_project_name,
                         hours_required_dict,
                         expertise_required_dict,
                         resource_names_list,
                         expertise_dict,
                         cost_dict,
                         days_list, # Pasar lista de números de día
                         availability_numeric # Pasar disponibilidad numérica
                     )

                st.session_state['last_run_success'] = optimal_cost is not None or bool(assignment) # Exito si hay coste o asignación
                st.session_state['optimal_cost'] = optimal_cost
                st.session_state['assignment'] = assignment

            except Exception as e:
                st.error(f"Ocurrió un error durante la ejecución del modelo: {e}")
                import traceback
                st.error(traceback.format_exc()) # Imprimir traceback completo para debug
                st.session_state['last_run_success'] = False
                st.session_state['optimal_cost'] = None
                st.session_state['assignment'] = None

        st.rerun()

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