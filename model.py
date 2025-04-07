# model.py
import pulp
from pyscipopt import Model, quicksum # Asegúrate de tener pyscipopt instalado: pip install pyscipopt
import time # Para medir tiempo si quieres

# --- Función con PuLP (CBC por defecto) ---
def solve_scheduling_problem(projects, tasks, hours_required, expertise_required,
                             resources, expertise, cost, days, availability):
    """Resuelve el problema de planificación usando PuLP (con correcciones)."""
    print("DEBUG: Iniciando solve_scheduling_problem (PuLP)") # <-- AÑADIDO (Debug)
    print(f"DEBUG: Recibidos {len(days)} días. Ejemplo availability[({resources[0]},{days[0]})]: {availability.get((resources[0], days[0]), 'No encontrado')}") # <-- AÑADIDO (Debug)


    level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
    model = pulp.LpProblem("ProjectScheduling_PuLP", pulp.LpMinimize)

    # Variables (sin cambios)
    x = pulp.LpVariable.dicts("assign", [(p, t, r) for p in projects for t in tasks[p] for r in resources], cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("tarea_dia_usado", [(p, t, d) for p in projects for t in tasks[p] for d in days], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("hours", [(p, t, r, d) for p in projects for t in tasks[p] for r in resources for d in days], lowBound=0, cat=pulp.LpContinuous)

    # Cálculo de penalización (sin cambios)
    max_cost_per_hour = max(cost.values()) if cost else 0
    total_hours = sum(hours_required.values()) if hours_required else 0
    max_possible_cost = max_cost_per_hour * total_hours if total_hours > 0 else 1000 # Evitar 0
    factor_penalizacion = max_possible_cost * 1.1 if max_possible_cost > 0 else 10000 # Evitar 0

    # Función objetivo (sin cambios)
    model += pulp.lpSum(cost[r] * y[(p, t, r, d)]
                        for p in projects for t in tasks[p] for r in resources for d in days) + \
             pulp.lpSum(z[(p, t, d)] for p in projects for t in tasks[p] for d in days) * factor_penalizacion, "Objetivo"

    # Restricciones
    # (Asignación única y expertise - sin cambios)
    for p in projects:
        for t in tasks[p]:
            model += pulp.lpSum(x[(p, t, r)] for r in resources) == 1, f"task_assigned_{p}_{t}"
            required_expertise_level = level_map[expertise_required[(p, t)]]
            for r in resources:
                if level_map[expertise[r]] < required_expertise_level:
                    model += x[(p, t, r)] == 0, f"expertise_{p}_{t}_{r}"
                # La restricción y <= 8*x se mantiene por ahora (BigM para y si x=1)
                for d in days:
                     model += y[(p, t, r, d)] <= x[(p, t, r)] * 8, f"BigM_hours_{p}_{t}_{r}_{d}" # M=8

            # Horas requeridas totales (sin cambios)
            model += pulp.lpSum(y[(p, t, r, d)] for r in resources for d in days) == hours_required[(p, t)], f"hours_required_{p}_{t}"

    # --- Disponibilidad de Recursos (CORREGIDO) ---
    for r in resources:
        for d in days:
            # Obtener horas disponibles del diccionario de entrada. Usa .get con default 0.
            # ASUME que 'availability' está indexado por (resource_name, day_number)
            available_hours_for_resource_day = availability.get((r, d), 0) # <-- MODIFICADO

            total_hours_assigned_to_resource_on_day = pulp.lpSum(
                y[(p, t, r, d)] for p in projects for t in tasks[p]
            )
            # La suma de horas no puede exceder la disponibilidad específica
            model += total_hours_assigned_to_resource_on_day <= available_hours_for_resource_day, f"avail_{r}_{d}" # <-- MODIFICADO

    # --- Vinculo y con z (usando M=8) ---
    M = 8 # <-- MODIFICADO (Usar M=8 directamente)
    for p in projects:
        for t in tasks[p]:
            # La restricción original estaba mal indexada (incluía 'r'). Corregido:
            # Si se trabaja en la tarea (p,t) en el dia d (sum(y) > 0), entonces z=1
            # Podemos hacer: sum(y[(p, t, r, d) for r in resources]) <= M_total_task_hours * z[(p, t, d)]
            # O la que tenías, que fuerza z=1 si CUALQUIER recurso trabaja en (p,t,d)
            # Mantendremos la que tenías pero con M=8 y asegurando la lógica
             for d in days:
                # Si cualquier recurso r trabaja en (p,t) en día d, z[(p,t,d)] debe ser 1
                # Lo hacemos por recurso para asegurar que no se exceda M si z=1
                for r in resources:
                     model += y[(p, t, r, d)] <= M * z[(p, t, d)], f"vinculo_y_z_{p}_{t}_{r}_{d}"


    # Resolver el modelo con límite de tiempo y logs
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300) # <-- MODIFICADO (msg=True, timeLimit=300s)
    print("Iniciando solver CBC...") # <-- AÑADIDO
    start_time = time.time()
    status = model.solve(solver)
    end_time = time.time()
    print(f"Solver CBC finalizado en {end_time - start_time:.2f} segundos.") # <-- AÑADIDO
    print(f"Estado del solver CBC: {pulp.LpStatus[status]}") # <-- MODIFICADO (más claro)

    optimal_cost = None
    assignment = {}
    # Asegurarse de que el estado es Óptimo o Factible (si el solver paró por tiempo)
    if status == pulp.LpStatusOptimal or status == pulp.LpStatusNotSolved: # Considerar NotSolved si hay timelimit
         try:
             optimal_cost = pulp.value(model.objective)
             print(f"Costo (potencialmente subóptimo si hubo timeout): {optimal_cost}") # <-- AÑADIDO
             for p in projects:
                 for t in tasks[p]:
                     for r in resources:
                         for d in days:
                             if pulp.value(y[(p, t, r, d)]) > 1e-4:
                                 assignment[(p, t, r, d)] = pulp.value(y[(p, t, r, d)])
         except Exception as e:
             print(f"Error al extraer la solución de PuLP: {e}")
             optimal_cost = None
             assignment = {}
             # Si el estado es NotSolved pero no hay solución, limpiamos
             if status == pulp.LpStatusNotSolved:
                 optimal_cost = None
                 assignment = {}


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
