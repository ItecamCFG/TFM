# model.py
from pyscipopt import Model, quicksum
import pulp
import time
from datetime import date, timedelta

# --- Función con PuLP (CBC) - v5.1 (Minimizar Makespan CORRECTO) ---
def solve_scheduling_problem(
    projects_list,              # Lista de diccionarios de proyecto (con 'id', 'name', 'deadline')
    tasks_list,                 # Lista de diccionarios de tarea (con 'id', 'project_id', 'name', 'hours', 'expertise', 'sequence')
    resource_names_list,        # Lista de nombres de recursos
    expertise_dict,             # {recurso: expertis}
    cost_dict,                  # {recurso: coste} - Aunque no se use en el objetivo, puede ser útil para info
    days_list,                  # Lista de números de día [1, 2, ..., N]
    availability_numeric,       # {(recurso, día_num): horas_disponibles}
    start_date                  # Fecha de inicio (date object) correspondiente al día 1
):
    """Resuelve el problema usando PuLP, minimizando makespan (v5.1)."""
    print("DEBUG: Iniciando solve_scheduling_problem (PuLP) - v5.1 (Min Makespan)")
    print(f"DEBUG: Recibidos {len(days_list)} días desde {start_date}. Disponibilidad({resource_names_list[0]},{days_list[0]}): {availability_numeric.get((resource_names_list[0], days_list[0]), 'N/A')}")

    # 1. Inicialización
    level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
    model = pulp.LpProblem("ProjectScheduling_PuLP_v5_Makespan", pulp.LpMinimize)

    # Mapeos y estructuras
    project_names = [p['name'] for p in projects_list]
    project_id_to_name = {p['id']: p['name'] for p in projects_list}
    hours_required_dict = {}
    expertise_required_dict = {}
    tasks_per_project_name = {p_name: [] for p_name in project_names}
    task_info = {}

    for task in tasks_list:
        proj_id = task['project_id']
        proj_name = project_id_to_name.get(proj_id)
        if proj_name:
            task_name = task['name']
            task_key = (proj_name, task_name)
            # Evitar duplicados por nombre si la estructura lo requiere
            if task_name not in tasks_per_project_name[proj_name]:
                tasks_per_project_name[proj_name].append(task_name)
                hours_required_dict[task_key] = task.get('hours', 0)
                expertise_required_dict[task_key] = task.get('expertise', "Junior")
                task_info[task_key] = task

    # Verificar que todas las tareas tengan horas > 0 si es un requisito
    for key, hours in hours_required_dict.items():
        if hours <= 0:
            print(f"ADVERTENCIA: Tarea {key} tiene {hours} horas requeridas. Esto puede causar problemas.")
            # hours_required_dict[key] = 0.1 # O manejarlo de otra forma

    # 2. Variables
    x = pulp.LpVariable.dicts("assign", [(p, t, r) for p in project_names for t in tasks_per_project_name[p] for r in resource_names_list], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("hours", [(p, t, r, d) for p in project_names for t in tasks_per_project_name[p] for r in resource_names_list for d in days_list], lowBound=0, cat=pulp.LpContinuous)

    # Variable binaria auxiliar: 1 si se trabaja en la tarea (p,t) el día d
    work_day = pulp.LpVariable.dicts("WorkDay", [(p, t, d) for p in project_names for t in tasks_per_project_name[p] for d in days_list], cat=pulp.LpBinary)

    # Variable de tiempo de finalización (puede ser continua o entera)
    # Usar continua puede ser más rápido, entera es más estricta. Probemos entera.
    end_day = pulp.LpVariable.dicts("EndDay", [(p, t) for p in project_names for t in tasks_per_project_name[p]], lowBound=1, cat=pulp.LpInteger)

    # Variable Makespan (objetivo)
    makespan = pulp.LpVariable("Makespan", lowBound=1, cat=pulp.LpInteger)

    # 3. Función Objetivo: Minimizar Makespan
    model += makespan, "Minimize_Makespan"

    # 4. Restricciones
    M_daily = 8 # Max horas razonable por tarea/recurso/día para BigM
    M_days = len(days_list) + 1 # Un día más que el horizonte para BigM

    # --- 4.1 Asignación única, Expertise, Horas Totales, Link y-x ---
    for p in project_names:
        for t in tasks_per_project_name[p]:
            task_key = (p, t)
            H_pt = hours_required_dict.get(task_key, 0) # Horas para esta tarea

            model += pulp.lpSum(x[(p, t, r)] for r in resource_names_list) == 1, f"assign_one_res_{p}_{t}"

            required_expertise_level = level_map.get(expertise_required_dict.get(task_key, "Junior"), 1)
            for r in resource_names_list:
                resource_expertise_level = level_map.get(expertise_dict.get(r, "Junior"), 1)
                if resource_expertise_level < required_expertise_level:
                    model += x[(p, t, r)] == 0, f"expertise_{p}_{t}_{r}"

                for d in days_list:
                    # Solo se pueden asignar horas si el recurso está asignado (y <= M*x)
                    model += y[(p, t, r, d)] <= M_daily * x[(p, t, r)], f"link_y_x_{p}_{t}_{r}_{d}"

            # Horas totales deben cumplirse
            model += pulp.lpSum(y[(p, t, r, d)] for r in resource_names_list for d in days_list) == H_pt, f"total_hours_{p}_{t}"

    # --- 4.2 Disponibilidad Diaria por Recurso ---
    for r in resource_names_list:
        for d in days_list:
            available_hours = availability_numeric.get((r, d), 0)
            model += pulp.lpSum(y[(p, t, r, d)] for p in project_names for t in tasks_per_project_name[p]) <= available_hours, f"avail_{r}_{d}"

    # --- 4.3 Enlace entre Trabajo (y), Días Activos (work_day) y Finalización (end_day) ---
    print("DEBUG: Añadiendo restricciones de enlace y-work_day-end_day...")
    for p in project_names:
        for t in tasks_per_project_name[p]:
            for d in days_list:
                # Si se trabaja alguna hora (sum(y)>0), work_day debe ser 1
                # Usamos y[p,t,r,d] <= M_daily * work_day[p,t,d] para cada recurso r
                # (Si CUALQUIER y > 0 para este (p,t,d), fuerza work_day=1)
                for r in resource_names_list:
                     model += y[(p, t, r, d)] <= M_daily * work_day[(p, t, d)], f"link_y_workday_{p}_{t}_{r}_{d}"

                # end_day debe ser >= a cualquier día 'd' en el que se trabaje (work_day=1)
                # end_day[p, t] >= d * work_day[p, t, d] (reescrito linealmente)
                # end_day[p, t] >= d - M_days * (1 - work_day[p, t, d])
                model += end_day[(p, t)] >= d - M_days * (1 - work_day[(p, t, d)]), f"link_endday_workday_{p}_{t}_{d}"
    print("DEBUG: Restricciones de enlace y-work_day-end_day añadidas.")


    # --- 4.4 Restricciones de Secuencialidad (Finish-to-Start usando end_day y work_day) ---
    print("DEBUG: Añadiendo restricciones de secuencia (Finish-to-Start)...")
    for p in project_names:
        sorted_tasks = sorted(
            tasks_per_project_name[p],
            key=lambda t: task_info.get((p, t), {}).get('sequence', float('inf'))
        )
        # Restricción entre tareas adyacentes t_i y t_{i+1}
        for i in range(len(sorted_tasks) - 1):
            t_i = sorted_tasks[i]
            t_i_plus_1 = sorted_tasks[i+1]

            # Opción 1: Simple End-Before-End (Débil pero simple)
            # model += end_day[(p, t_i)] <= end_day[(p, t_i_plus_1)], f"seq_weak_{p}_{t_i}_{t_i_plus_1}"

            # Opción 2: Finish-to-Start (Más estricta):
            # La tarea t_{i+1} no puede estar activa (work_day=1) en un día 'd'
            # a menos que la tarea t_i haya finalizado el día anterior (end_day[t_i] <= d-1)
            # Lo formulamos con Big-M: end_day[t_i] <= (d - 1) + M_days * (1 - work_day[t_{i+1}, d])
            # Esto significa: si work_day[t_{i+1}, d] = 1 => end_day[t_i] <= d - 1
            for d in days_list:
                 if d > 1: # No aplica para el primer día
                     model += end_day[(p, t_i)] <= (d - 1) + M_days * (1 - work_day[(p, t_i_plus_1, d)]), f"seq_finish_start_{p}_{t_i}_{t_i_plus_1}_day_{d}"
                 else: # Para d=1, work_day[t_{i+1}, 1] solo puede ser 1 si t_i requiere 0 horas (no debería pasar)
                      # Si t_i tiene horas > 0, no puede terminar antes del día 1.
                      # Podemos forzar a que si t_i tiene horas, t_{i+1} no pueda estar activo el día 1.
                      if hours_required_dict.get((p,t_i),0) > 0:
                           model += work_day[(p, t_i_plus_1, 1)] == 0, f"seq_no_start_d1_{p}_{t_i}_{t_i_plus_1}"


    print("DEBUG: Restricciones de secuencia (Finish-to-Start) añadidas.")

    # --- 4.5 Restricciones de Deadline (Finalizar antes o en el deadline) ---
    print("DEBUG: Añadiendo restricciones de deadline...")
    for project_data in projects_list:
        p_name = project_data['name']
        deadline_date = project_data.get('deadline') # Es un objeto date
        if deadline_date:
            deadline_day_number = (deadline_date - start_date).days + 1
            # Asegurar que el deadline esté dentro del horizonte considerado
            if deadline_day_number <= days_list[-1]:
                for t in tasks_per_project_name[p_name]:
                    # end_day[p, t] debe ser <= deadline_day_number
                    model += end_day[(p_name, t)] <= deadline_day_number, f"deadline_{p_name}_{t}"
            else:
                print(f"ADVERTENCIA: Deadline para proyecto {p_name} ({deadline_date}) está fuera del horizonte de planificación ({days_list[-1]} días desde {start_date}). No se aplicará restricción.")
    print("DEBUG: Restricciones de deadline añadidas.")

    # --- 4.6 Definición del Makespan ---
    print("DEBUG: Añadiendo definición de makespan...")
    for p in project_names:
        for t in tasks_per_project_name[p]:
            # makespan debe ser >= que el tiempo de finalización de cualquier tarea
            model += makespan >= end_day[(p, t)], f"makespan_def_{p}_{t}"
    print("DEBUG: Definición de makespan añadida.")

    # 5. Resolver
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300) # Mantener límite y logs
    print("Iniciando solver CBC...")
    start_time = time.time()
    status = model.solve(solver)
    end_time = time.time()
    solver_runtime = end_time - start_time
    print(f"Solver CBC finalizado en {solver_runtime:.2f} segundos.")
    print(f"Estado del solver CBC: {pulp.LpStatus[status]}")

    # 6. Extraer Resultados
    optimal_makespan = None
    assignment = {}
    task_completion_days = {} # Cambiado nombre para claridad

    # Extraer la solución si es Óptima o si se alcanzó el límite de tiempo pero se encontró *alguna* solución
    if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and model.objective is not None and pulp.value(model.objective) is not None):
         try:
             optimal_makespan = pulp.value(makespan)
             print(f"Makespan óptimo (o subóptimo si hubo timeout): {optimal_makespan}")
             # Extraer días de finalización
             for p in project_names:
                 for t in tasks_per_project_name[p]:
                     task_completion_days[(p, t)] = pulp.value(end_day[(p, t)])
             # Extraer asignación de horas
             for p in project_names:
                 for t in tasks_per_project_name[p]:
                     for r in resource_names_list:
                         for d in days_list:
                             var_value = pulp.value(y[(p, t, r, d)])
                             # Solo guardar si hay horas asignadas (> epsilon)
                             if var_value is not None and var_value > 1e-4:
                                 assignment[(p, t, r, d)] = var_value
         except Exception as e:
             print(f"Error al extraer la solución de PuLP: {e}")
             optimal_makespan = None
             assignment = {}
             task_completion_days = {}
             # Limpiar si el estado es NotSolved pero no se pudo extraer
             if status == pulp.LpStatusNotSolved:
                 optimal_makespan = None
                 assignment = {}
                 task_completion_days = {}

    if optimal_makespan is None:
         print("No se encontró/extrajo una solución factible para el makespan.")

    # Devolver makespan, asignación y tiempos de finalización
    return optimal_makespan, assignment, task_completion_days


# --- Función con PySCIPOpt ---
def solve_scheduling_problem_scip(projects, tasks, hours_required, expertise_required,
                                  resources, expertise, cost, days, availability):
    """Resuelve el problema usando PySCIPOpt (con correcciones)."""
    print("DEBUG: Iniciando solve_scheduling_problem_scip") # <-- AÑADIDO (Debug)
    print(f"DEBUG: Recibidos {len(days)} días. Ejemplo availability[({resources[0]},{days[0]})]: {availability.get((resources[0], days[0]), 'No encontrado')}") # <-- AÑADIDO (Debug)

    level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
    model = Model("ProjectSchedulingSCIP")

    # 1) VARIABLES (igual, M=8)
    x, y, z = {}, {}, {}
    M = 8 # <-- MODIFICADO
    for p in projects:
        for t in tasks[p]:
            for r in resources:
                x[(p, t, r)] = model.addVar(vtype="B", name=f"x_{p}_{t}_{r}")
                for d in days:
                    y[(p, t, r, d)] = model.addVar(lb=0, ub=M, vtype="C", name=f"y_{p}_{t}_{r}_{d}")
            for d in days:
                 z[(p, t, d)] = model.addVar(vtype="B", name=f"z_{p}_{t}_{d}")


    # 2) FUNCIÓN OBJETIVO (igual)
    max_cost_per_hour = max(cost.values()) if cost else 0
    total_hours = sum(hours_required.values()) if hours_required else 0
    max_possible_cost = max_cost_per_hour * total_hours if total_hours > 0 else 1000
    factor_penalizacion = max_possible_cost * 1.1 if max_possible_cost > 0 else 10000
    objective = quicksum(cost[r] * y[(p, t, r, d)] for p in projects for t in tasks[p] for r in resources for d in days) + \
                factor_penalizacion * quicksum(z[(p, t, d)] for p in projects for t in tasks[p] for d in days)
    model.setObjective(objective, "minimize")

    # 3) RESTRICCIONES
    # (a) Asignación única (igual)
    for p in projects:
        for t in tasks[p]:
            model.addCons(quicksum(x[(p, t, r)] for r in resources) == 1, name=f"task_assigned_{p}_{t}")

    # (b) Expertise (igual)
    for p in projects:
        for t in tasks[p]:
            required_expertise_level = level_map[expertise_required[(p, t)]]
            for r in resources:
                if level_map[expertise[r]] < required_expertise_level:
                    model.addCons(x[(p, t, r)] == 0, name=f"expertise_{p}_{t}_{r}")

    # (c) Horas requeridas totales (igual)
    for p in projects:
        for t in tasks[p]:
            model.addCons(quicksum(y[(p, t, r, d)] for r in resources for d in days) == hours_required[(p, t)], name=f"hours_required_{p}_{t}")

    # (d) Vincular y con x: y <= M*x (igual, M=8)
    for p in projects:
        for t in tasks[p]:
            for r in resources:
                for d in days:
                    model.addCons(y[(p, t, r, d)] <= M * x[(p, t, r)], name=f"BigM_hours_{p}_{t}_{r}_{d}")

    # --- (e) Disponibilidad diaria (CORREGIDO) ---
    for r in resources:
        for d in days:
            # Obtener horas disponibles del diccionario de entrada. Usa .get con default 0.
            # ASUME que 'availability' está indexado por (resource_name, day_number)
            available_hours_for_resource_day = availability.get((r, d), 0) # <-- MODIFICADO

            total_hours_on_day = quicksum(y[(p, t, r, d)] for p in projects for t in tasks[p])
            # La suma de horas no puede exceder la disponibilidad específica
            model.addCons(total_hours_on_day <= available_hours_for_resource_day, name=f"avail_{r}_{d}") # <-- MODIFICADO

    # (f) Vincular y con z: y[(p,t,r,d)] <= M * z[(p,t,d)] (igual, M=8)
    for p in projects:
        for t in tasks[p]:
            for d in days:
                 for r in resources: # Asegurar que si CUALQUIER recurso trabaja, z=1
                    model.addCons(y[(p, t, r, d)] <= M * z[(p, t, d)], name=f"vinculo_y_z_{p}_{t}_{r}_{d}")


    # 4) RESOLVER con Límite de Tiempo y Logs Visibles
    # model.hideOutput() # <-- MODIFICADO (Comentado para ver logs)
    model.setParam('limits/time', 300) # <-- AÑADIDO (300 segundos = 5 minutos)
    print("Iniciando solver SCIP...") # <-- AÑADIDO
    start_time = time.time()
    try:
        model.optimize()
    except Exception as e:
        print(f"Error durante model.optimize(): {e}") # Capturar posibles errores de SCIP
    end_time = time.time()
    print(f"Solver SCIP finalizado en {end_time - start_time:.2f} segundos.") # <-- AÑADIDO
    status = model.getStatus()
    print(f"Estado del solver SCIP: {status}") # <-- MODIFICADO (más claro)

    # 5) EXTRAER SOLUCIÓN
    optimal_cost = None
    assignment = {}
    # Extraer solución si es óptima O si se encontró una solución factible antes del timeout
    if status == "optimal" or (status == "timelimit" and model.getNSols() > 0):
        try:
            optimal_cost = model.getObjVal()
            print(f"Costo (potencialmente subóptimo si hubo timeout): {optimal_cost}") # <-- AÑADIDO

        #------------ GetSolVal ------------------
            best_solution = model.getBestSol()
            if best_solution is not None:
            # Guardar solo las horas > 0.0001
                for p in projects:
                    for t in tasks[p]:
                        for r in resources:
                            for d in days:
                                # Usar model.getSolVal() para obtener valor de la mejor solución encontrada
                                val = model.getSolVal(best_solution,y[(p, t, r, d)])
                                if val > 1e-4:
                                    assignment[(p, t, r, d)] = val
        except Exception as e:
            # A veces SCIP puede dar error al extraer la solución si no terminó bien
             print(f"Error al extraer la solución de SCIP: {e}")
             optimal_cost = None
             assignment = {}
    # Si no hay solución o hubo error antes
    if optimal_cost is None:
         print("No se encontró solución óptima o factible, o hubo un error.")

    return optimal_cost, assignment


if __name__ == '__main__':
    # Ejemplo de uso (para pruebas)
    projects_data = [{'id': 1, 'name': 'Project Alpha', 'deadline': 5},
                     {'id': 2, 'name': 'Project Beta', 'deadline': 10}]
    tasks_data = [{'id': 1, 'project_id': 1, 'name': 'Analysis', 'hours': 10, 'expertise': 'Junior', 'sequence': 1},
                  {'id': 2, 'project_id': 1, 'name': 'Development', 'hours': 20, 'expertise': 'Senior', 'sequence': 2},
                  {'id': 3, 'project_id': 2, 'name': 'Design', 'hours': 15, 'expertise': 'Expert', 'sequence': 1}]
    hours_required = {('Project Alpha', 'Analysis'): 10, ('Project Alpha', 'Development'): 20, ('Project Beta', 'Design'): 15}
    expertise_required = {('Project Alpha', 'Analysis'): 'Junior', ('Project Alpha', 'Development'): 'Senior', ('Project Beta', 'Design'): 'Experto'}
    resources_data = [{'name': 'Alice', 'expertise': 'Senior', 'cost': 50, 'Lunes': 8, 'Martes': 8, 'Miércoles': 8, 'Jueves': 8, 'Viernes': 8},
                      {'name': 'Bob', 'expertise': 'Junior', 'cost': 30, 'Lunes': 8, 'Martes': 8, 'Miércoles': 8, 'Jueves': 8, 'Viernes': 8},
                      {'name': 'Charlie', 'expertise': 'Experto', 'cost': 70, 'Lunes': 8, 'Martes': 8, 'Miércoles': 8, 'Jueves': 8, 'Viernes': 8}]
    resource_names = [r['name'] for r in resources_data]
    expertise = {r['name']: r['expertise'] for r in resources_data}
    cost = {r['name']: r['cost'] for r in resources_data}
    days = list(range(1, 15))
    availability = {}
    for r in resources_data:
        for d in days:
            weekday = (d - 1) % 7
            day_name = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][weekday]
            availability[(r['name'], d)] = r.get(day_name, 0) if weekday < 5 else 0

    optimal_cost, assignment = solve_scheduling_problem(projects_data, tasks_data, hours_required, expertise_required,resource_names, expertise, cost, days, availability)

    if optimal_cost is not None:
        print(f"Optimal cost: {optimal_cost}")
        for key, value in assignment.items():
            print(f"{key}: {value}")
    else:
        print("No solution found or error occurred.")
