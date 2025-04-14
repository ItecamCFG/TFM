# model.py
from pyscipopt import Model, quicksum
import pulp
import time
from datetime import date, timedelta # Asegúrate de importar date y timedelta

# --- Función con PuLP (CBC) - Versión Simplificada + Secuencia + Deadline ---
def solve_scheduling_problem(
    projects_list,              # Lista de diccionarios de proyecto (con 'id', 'name', 'deadline')
    tasks_list,                 # Lista de diccionarios de tarea (con 'id', 'project_id', 'name', 'hours', 'expertise', 'sequence')
    resource_names_list,        # Lista de nombres de recursos
    expertise_dict,             # {recurso: expertis}
    cost_dict,                  # {recurso: coste}
    days_list,                  # Lista de números de día [1, 2, ..., N]
    availability_numeric,       # {(recurso, día_num): horas_disponibles}
    start_date= date.today()    # Fecha de inicio (date object) correspondiente al día 1 de days_list <-- NUEVO ARGUMENTO
    ):
    """Resuelve el problema usando PuLP, minimizando coste y añadiendo secuencia/deadline."""
    print("DEBUG: Iniciando solve_scheduling_problem (PuLP) - v4 (Secuencia/Deadline)")
    print(f"DEBUG: Recibidos {len(days_list)} días desde {start_date}. Ejemplo availability[({resource_names_list[0]},{days_list[0]})]: {availability_numeric.get((resource_names_list[0], days_list[0]), 'No encontrado')}")

    # 1. Inicialización
    level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
    model = pulp.LpProblem("ProjectScheduling_PuLP_v4", pulp.LpMinimize)

    # Mapeos y estructuras de datos necesarias
    project_names = [p['name'] for p in projects_list]
    project_id_to_name = {p['id']: p['name'] for p in projects_list}
    # Necesitamos hours_required y expertise_required por (proj_name, task_name)
    hours_required_dict = {}
    expertise_required_dict = {}
    tasks_per_project_name = {p_name: [] for p_name in project_names}
    task_info = {} # Guardar info completa de tarea por (proj_name, task_name)

    for task in tasks_list:
        proj_id = task['project_id']
        proj_name = project_id_to_name.get(proj_id)
        if proj_name:
            task_name = task['name']
            task_key = (proj_name, task_name)
            if task_name not in tasks_per_project_name[proj_name]:
                tasks_per_project_name[proj_name].append(task_name)
                hours_required_dict[task_key] = task['hours']
                expertise_required_dict[task_key] = task['expertise']
                task_info[task_key] = task # Guardar secuencia, etc.

    # 2. Variables
    # x_p,t,r : 1 si recurso r asignado a tarea (p,t)
    # y_p,t,r,d: Horas trabajadas por r en (p,t) el día d
    x = pulp.LpVariable.dicts("assign",
                               [(p, t, r) for p in project_names for t in tasks_per_project_name[p] for r in resource_names_list],
                               cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("hours",
                               [(p, t, r, d) for p in project_names for t in tasks_per_project_name[p] for r in resource_names_list for d in days_list],
                               lowBound=0, cat=pulp.LpContinuous)
    # La variable z ya no es necesaria

    # 3. Función Objetivo: Minimizar Coste Total
    model += pulp.lpSum(cost_dict[r] * y[(p, t, r, d)]
                        for p in project_names for t in tasks_per_project_name[p] for r in resource_names_list for d in days_list), "Total_Cost"

    # 4. Restricciones
    M_daily = 8 # Máximo de horas diarias asumido para BigM

    # --- 4.1 Asignación única y Expertise ---
    for p in project_names:
        for t in tasks_per_project_name[p]:
            task_key = (p, t)
            # Asignar exactamente un recurso
            model += pulp.lpSum(x[(p, t, r)] for r in resource_names_list) == 1, f"task_assigned_{p}_{t}"
            # Verificar expertise
            required_expertise_level = level_map.get(expertise_required_dict.get(task_key, "Junior"), 1)
            for r in resource_names_list:
                resource_expertise_level = level_map.get(expertise_dict.get(r, "Junior"), 1)
                if resource_expertise_level < required_expertise_level:
                    model += x[(p, t, r)] == 0, f"expertise_{p}_{t}_{r}"
                # Vincular y con x: solo puede trabajar si está asignado (y <= M*x)
                # Y limitar horas diarias por tarea/recurso si se desea (opcional pero puede ayudar)
                for d in days_list:
                    # Limita las horas en esta tarea específica por este recurso en este día a M_daily *si está asignado*
                    model += y[(p, t, r, d)] <= M_daily * x[(p, t, r)], f"link_y_x_{p}_{t}_{r}_{d}"

            # Cumplir horas totales requeridas para la tarea
            model += pulp.lpSum(y[(p, t, r, d)] for r in resource_names_list for d in days_list) == hours_required_dict[task_key], f"hours_required_{p}_{t}"

    # --- 4.2 Disponibilidad Diaria por Recurso ---
    for r in resource_names_list:
        for d in days_list:
            available_hours = availability_numeric.get((r, d), 0)
            model += pulp.lpSum(y[(p, t, r, d)] for p in project_names for t in tasks_per_project_name[p]) <= available_hours, f"avail_{r}_{d}"

    # --- 4.3 Restricciones de Secuencialidad (Fracción Completada) ---
    print("DEBUG: Añadiendo restricciones de secuencia...")
    for p in project_names:
        # Obtener tareas ordenadas por secuencia para este proyecto
        sorted_tasks = sorted(
            tasks_per_project_name[p],
            key=lambda t: task_info.get((p, t), {}).get('sequence', float('inf'))
        )

        # Añadir restricción entre tareas adyacentes t_i y t_{i+1}
        for i in range(len(sorted_tasks) - 1):
            t_i = sorted_tasks[i]
            t_i_plus_1 = sorted_tasks[i+1]
            task_key_i = (p, t_i)
            task_key_i_plus_1 = (p, t_i_plus_1)

            H_i = hours_required_dict.get(task_key_i, 1) # Usar 1 si horas es 0 para evitar división por cero
            H_i_plus_1 = hours_required_dict.get(task_key_i_plus_1, 1)
            if H_i <= 0: H_i = 1 # Evitar división por cero
            if H_i_plus_1 <= 0: H_i_plus_1 = 1

            # Para cada día d, la fracción completada de t_{i+1} no puede superar la de t_i
            for d in days_list:
                cumulative_y_i = pulp.lpSum(y[(p, t_i, r, k)] for r in resource_names_list for k in days_list if k <= d)
                cumulative_y_i_plus_1 = pulp.lpSum(y[(p, t_i_plus_1, r, k)] for r in resource_names_list for k in days_list if k <= d)

                # (Suma_y_{i+1} / H_{i+1}) <= (Suma_y_i / H_i)
                # Reordenado para evitar división por variable si H fuera variable:
                # Suma_y_{i+1} * H_i <= Suma_y_i * H_{i+1}
                model += cumulative_y_i_plus_1 * H_i <= cumulative_y_i * H_i_plus_1, f"seq_{p}_{t_i}_{t_i_plus_1}_day_{d}"
    print("DEBUG: Restricciones de secuencia añadidas.")

    # --- 4.4 Restricciones de Deadline (No trabajar después del deadline) ---
    print("DEBUG: Añadiendo restricciones de deadline...")
    for project_data in projects_list:
        p_name = project_data['name']
        deadline_date = project_data.get('deadline') # Es un objeto date

        if deadline_date:
            # Calcular el número de día correspondiente al deadline
            # Sumamos 1 porque days_list empieza en 1
            deadline_day_number = (deadline_date - start_date).days + 1

            # Encontrar los días en nuestra planificación que son ESTRICTAMENTE posteriores al deadline
            days_after_deadline = [d for d in days_list if d > deadline_day_number]

            if days_after_deadline:
                # Forzar a que no haya horas asignadas en esos días para NINGUNA tarea de este proyecto
                for t in tasks_per_project_name[p_name]:
                     model += pulp.lpSum(y[(p_name, t, r, d)] for r in resource_names_list for d in days_after_deadline) == 0, f"deadline_{p_name}_{t}"
                     # print(f"DEBUG: Restricción deadline para {p_name}-{t} -> no trabajar en días > {deadline_day_number} ({days_after_deadline})")

    print("DEBUG: Restricciones de deadline añadidas.")


    # 5. Resolver
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300) # Mantener límite y logs
    print("Iniciando solver CBC...")
    start_time = time.time()
    status = model.solve(solver)
    end_time = time.time()
    print(f"Solver CBC finalizado en {end_time - start_time:.2f} segundos.")
    print(f"Estado del solver CBC: {pulp.LpStatus[status]}")

    # 6. Extraer Resultados (igual que antes)
    optimal_cost = None
    assignment = {}
    if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and pulp.value(model.objective) is not None) : # Si hay timeout pero encontró algo
         try:
             optimal_cost = pulp.value(model.objective)
             print(f"Costo (potencialmente subóptimo si hubo timeout): {optimal_cost}")
             for p in project_names:
                 for t in tasks_per_project_name[p]:
                     for r in resource_names_list:
                         for d in days_list:
                             var_value = pulp.value(y[(p, t, r, d)])
                             if var_value is not None and var_value > 1e-4:
                                 assignment[(p, t, r, d)] = var_value
         except Exception as e:
             print(f"Error al extraer la solución de PuLP: {e}")
             optimal_cost = None
             assignment = {}
             if status == pulp.LpStatusNotSolved:
                 optimal_cost = None
                 assignment = {}

    if not assignment:
         print("No se encontró/extrajo una asignación factible.")
         optimal_cost = None

    return optimal_cost, assignment


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
