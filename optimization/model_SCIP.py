# optimization/makespan_model_scip.py
import time
# Necesitas instalar PySCIPOpt: pip install pyscipopt
try:
    # Importar SCIP_STATUS. Si TIMEOUT da error, es un problema local de la instalacion
    from pyscipopt import Model, quicksum, SCIP_STATUS # Eliminamos TIMEOUT de la importación directa si es necesario
    # Dependiendo de la version, TIMEOUT puede ser accesible directamente o no.
    # La traza sugiere que el problema es el atributo en el objeto SCIP_STATUS, no la importacion del nombre.

except ImportError:
    # Proporcionar un mensaje útil si PySCIPOpt no está instalado
    raise ImportError(
        "La biblioteca 'pyscipopt' no está instalada. "
        "Por favor, instálala usando: pip install pyscipopt"
    )

from optimization.base_model import OptimizationModel
from optimization.data_models import OptimizationResult  # Importar clase resultado
from datetime import date, timedelta

# Custom event handler (mantenemos por si acaso, aunque no es relevante para este error)
# class SolutionPrinter(Eventhdlr):
#     def __init__(self, model):
#         self.model = model
#         self.solution_count = 0
#
#     def eventexec(self, event):
#         if event == SCIP_EVENTTYPE.BESTSOLFOUND:
#             self.solution_count += 1
#             pass # Keep it silent for now


class MakespanMinimizationSCIP(OptimizationModel):
    """Minimiza Makespan usando SCIP a través de PySCIPOpt."""

    # Constructor de la clase padre
    def __init__(self, input_data):
        print("DEBUG (Init): Inicializando modelo SCIP...")
        super().__init__(input_data)

    def _build_model(self):
        """Construye el modelo SCIP para minimizar makespan."""
        self.model = Model("Makespan_SCIP")
        self.variables = {} # Reiniciar variables para este modelo

        # --- Definir Variables ---
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8
        # self.days_list y self.M_days should be available from base class _prepare_common_data
        M_days = getattr(self, 'M_days', len(getattr(self, 'days_list', [0])) + 1)


        # x_p,t,r : 1 si recurso r asignado a tarea (p,t)
        # y_p,t,r,d: Horas trabajadas por r en (p,t) el día d
        # work_day_p,t,d: 1 si se trabaja en (p,t) el día d
        # end_day_p,t: Día de finalización de la tarea (p,t)
        # makespan: Día de finalización del último proyecto

        x = {}
        y = {}
        work_day = {}
        end_day = {}

        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                for r in self.resource_names:
                    x[(p, t, r)] = self.model.addVar(vtype="B", name=f"assign_{p}_{t}_{r}")
                for d in self.days_list:
                    for r in self.resource_names:
                        y[(p, t, r, d)] = self.model.addVar(vtype="C", name=f"hours_{p}_{t}_{r}_{d}", lb=0)
                    work_day[(p, t, d)] = self.model.addVar(vtype="B", name=f"WorkDay_{p}_{t}_{d}")
                end_day[(p, t)] = self.model.addVar(vtype="I", name=f"EndDay_{p}_{t}", lb=1)


        makespan = self.model.addVar(vtype="I", name="Makespan", lb=1)

        # Guardar referencias
        self.variables = {'x': x, 'y': y, 'work_day': work_day, 'end_day': end_day, 'makespan': makespan}

        # --- Objetivo ---
        self.model.setObjective(makespan, "minimize")

        # --- Restricciones ---
        # 4.1 Asignación única y validación de expertise
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                self.model.addCons(
                    quicksum(x[(p, t, r)] for r in self.resource_names) == 1,
                    f"assign_one_res_{p}_{t}"
                )

                # self.expertise_required_dict should be available from base class _prepare_common_data
                required_level = level_map.get(self.expertise_required_dict.get((p, t), "Junior"), 1)
                for r in self.resource_names:
                    # self.expertise_dict should be available from base class _prepare_common_data
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < required_level:
                        self.model.addCons(x[(p, t, r)] == 0, f"expertise_block_{p}_{t}_{r}")

                    for d in self.days_list:
                         self.model.addCons(y[(p, t, r, d)] <= M_daily * x[(p, t, r)], f"link_y_x_{p}_{t}_{r}_{d}")


                # 4.1 Horas totales por tarea
                # self.hours_required_dict should be available from base class _prepare_common_data
                self.model.addCons(
                    quicksum(y[(p, t, r, d)] for r in self.resource_names for d in self.days_list) == self.hours_required_dict[(p, t)],
                    f"total_hours_{p}_{t}"
                )


        # 4.2 Disponibilidad diaria de recursos
        for r in self.resource_names:
            for d in self.days_list:
                # self.availability_numeric should be available from base class _prepare_common_data
                self.model.addCons(
                    quicksum(y[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name[p]) <= self.availability_numeric.get((r, d), 0),
                    f"avail_{r}_{d}"
                )

        # 4.3 Enlace entre horas (y), días activos (work_day) y día de finalización (end_day)
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                for d in self.days_list:
                    # Link y to work_day: if any work is done, work_day is 1
                    self.model.addCons(
                        quicksum(y[(p, t, r, d)] for r in self.resource_names) <= M_daily * work_day[(p, t, d)],
                        f"link_y_work_upper_{p}_{t}_{d}"
                    )
                    # Link work_day to y: if work_day is 1, total hours on that day must be > epsilon
                    epsilon = 0.01 # small value
                    self.model.addCons(
                         quicksum(y[(p, t, r, d)] for r in self.resource_names) >= epsilon * work_day[(p, t, d)],
                         f"link_work_y_lower_{p}_{t}_{d}"
                    )


                    # Link end_day to work_day: end_day must be >= the last day work was done
                    self.model.addCons(
                         end_day[(p, t)] >= d - M_days * (1 - work_day[(p, t, d)]),
                         f"link_endday_work_{p}_{t}_{d}"
                    )

        # 4.4 Restricciones de secuencia entre tareas (Finish to Start)
        print("DEBUG (Build): Añadiendo restricciones de secuencia...")
        for p in self.project_names:
            # self.task_info should be available from base class _prepare_common_data
            sorted_tasks = sorted(
                self.tasks_per_project_name[p],
                key=lambda t: self.task_info[(p, t)].sequence if (p, t) in self.task_info and hasattr(self.task_info.get((p, t)), 'sequence') else float('inf')
            )
            for i in range(len(sorted_tasks) - 1):
                t_i = sorted_tasks[i]
                t_next = sorted_tasks[i+1]
                # The next task cannot start until the current task is finished.
                # Using end_day: end_day(t_i) <= start_day(t_next) - 1
                # Defining start_day can be complex. A simpler approach using end_day and work_day:
                # If work is done on t_next on day d, then t_i must have finished by day d-1.
                for d in self.days_list:
                     if d > 1:
                         self.model.addCons(
                             end_day[(p, t_i)] <= (d - 1) + M_days * (1 - work_day[(p, t_next, d)]),
                             f"seq_{p}_{t_i}_{t_next}_d{d}"
                         )
                     else: # Day 1
                         # self.hours_required_dict should be available from base class _prepare_common_data
                         if self.hours_required_dict.get((p, t_i), 0) > 0: # Only if the previous task actually requires hours
                            self.model.addCons(
                                work_day[(p, t_next, 1)] == 0, # Task t_next cannot start on day 1 if t_i exists and has hours
                                f"seq_d1_{p}_{t_i}_{t_next}"
                            )


        # 4.5 Restricciones de deadline (si existen)
        # self.input_data.projects and self.start_date should be available from base class
        for project in self.input_data.projects:
            if project.deadline:
                # Ensure self.start_date is a date object
                if not isinstance(self.start_date, date):
                     print(f"⚠️ start_date no es un objeto date para proyecto {project.name}")
                     continue # Skip this deadline constraint if start_date is invalid

                deadline_day = (project.deadline - self.start_date).days + 1
                if deadline_day > 0 and deadline_day <= self.days_list[-1]:
                     for t in self.tasks_per_project_name[project.name]:
                        self.model.addCons(
                            end_day[(project.name, t)] <= deadline_day,
                            f"deadline_{project.name}_{t}"
                        )
                elif deadline_day <= 0:
                     print(f"⚠️ Deadline en el pasado para proyecto {project.name}: {project.deadline}")
                else:
                     print(f"⚠️ Deadline fuera del horizonte ({self.days_list[-1]} días) para proyecto {project.name}: {project.deadline}")


        # 4.6 Definición de Makespan
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                self.model.addCons(
                    makespan >= end_day[(p, t)],
                    f"makespan_def_{p}_{t}"
                )

        print("DEBUG (Build): Modelo SCIP construido.")


    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el modelo SCIP construido."""
        if self.model is None:
            return "Error building model", 0.0

        # Configurar el solver (tiempo límite)
        if self.input_data.config.solver_time_limit:
             self.model.setParam('limits/time', self.input_data.config.solver_time_limit)

        # Add event handler (optional, commented out for now)
        # try:
        #     event_handler = SolutionPrinter(self.model)
        #     self.model.includeEventhdlr(event_handler, "SolutionPrinter", "prints solutions")
        #     from pyscipopt import SCIP_EVENTTYPE # Import here if needed
        #     self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, event_handler)
        # except ImportError:
        #      print("Warning: Could not import SCIP_EVENTTYPE for event handler.")
        # except Exception as e:
        #      print(f"Warning: Could not include event handler: {e}")


        start_t = time.time()
        self.model.optimize()
        runtime = time.time() - start_t

        # Map SCIP status to string
        # Get the status object
        status = self.model.getStatus()
        status_str = "Unknown" # Default status string

        # Use a more robust way to map status, checking for attribute existence
        if status == SCIP_STATUS.OPTIMAL:
            status_str = "Optimal"
        elif status == SCIP_STATUS.INFEASIBLE:
            status_str = "Infeasible"
        elif status == SCIP_STATUS.UNBOUNDED:
            status_str = "Unbounded"
        elif status == SCIP_STATUS.INFORUNBD:
            status_str = "Infeasible or Unbounded"
        # Check for TIMEOUT attribute existence before using it
        elif hasattr(SCIP_STATUS, 'TIMEOUT') and status == SCIP_STATUS.TIMEOUT:
             status_str = "Timeout"
        elif status == SCIP_STATUS.USERINTERRUPT:
            status_str = "User Interrupt"
        elif status == SCIP_STATUS.NODELIMIT:
            status_str = "Node Limit Exceeded"
        elif status == SCIP_STATUS.TOTALNODELIMIT:
            status_str = "Total Node Limit Exceeded"
        elif status == SCIP_STATUS.STALLNODELIMIT:
            status_str = "Stall Node Limit Exceeded"
        elif status == SCIP_STATUS.MEMLIMIT:
            status_str = "Memory Limit Exceeded"
        elif status == SCIP_STATUS.GAPLIMIT:
            status_str = "Gap Limit Reached"
        elif status == SCIP_STATUS.SOLLIMIT:
            status_str = "Solution Limit Reached"
        elif status == SCIP_STATUS.BESTSOLLIMIT:
            status_str = "Best Solution Limit Reached"
        elif status == SCIP_STATUS.RESTARTLIMIT:
            status_str = "Restart Limit Exceeded"
        elif status == SCIP_STATUS.UNKNOWN:
            status_str = "Unknown"
        else:
            # Fallback for any other status, try to get the name if available
             try:
                 status_str = status.name
             except:
                 status_str = f"Unknown Status Code: {status}"


        # Check if a solution was found even if not optimal (e.g., timeout, limit)
        if status != SCIP_STATUS.OPTIMAL and self.model.getNSols() > 0:
             if hasattr(SCIP_STATUS, 'TIMEOUT') and status == SCIP_STATUS.TIMEOUT:
                  status_str = "Timeout (Solution Found)"
             else:
                  status_str = f"Solution Found ({status_str})" # Indicate a solution was found


        return status_str, runtime

    def _extract_results(self, status: str):
        """Extrae los resultados del modelo SCIP resuelto."""
        makespan_val = None
        assignment = {}
        task_completion_days = {}
        objective_val = None

        # Check if a solution exists
        if self.model and self.model.getNSols() > 0:
             try:
                 # Get the best solution found
                 best_sol = self.model.getBestSol()

                 # Check if the best solution is valid (sometimes getBestSol can return None)
                 if best_sol:
                    objective_val = self.model.getObjVal() # Get objective of the best found solution
                    # Check if makespan variable exists and has a value in the solution
                    makespan_var = self.variables.get('makespan')
                    if makespan_var and self.model.getSolVal(best_sol, makespan_var) is not None:
                        makespan_val = self.model.getSolVal(best_sol, makespan_var)
                    else:
                         # If makespan variable value is not available, try to derive from task end days
                         if task_completion_days: # Assuming task_completion_days is populated below
                             makespan_val = max(task_completion_days.values()) if task_completion_days else None
                         else:
                             print("Warning: Makespan variable value not available and could not derive from task end days.")


                    y = self.variables['y']
                    end_day = self.variables['end_day']

                    # Extraer tiempos de finalización de tareas
                    for p in self.project_names:
                       for t in self.tasks_per_project_name[p]:
                           end_day_var = end_day.get((p, t))
                           if end_day_var and self.model.getSolVal(best_sol, end_day_var) is not None:
                                # Round to nearest integer day as end_day is integer variable
                                task_completion_days[(p, t)] = round(self.model.getSolVal(best_sol, end_day_var))


                    # Extraer asignaciones (horas trabajadas)
                    for p in self.project_names:
                       for t in self.tasks_per_project_name[p]:
                           for r in self.resource_names:
                               for d in self.days_list:
                                   y_var = y.get((p, t, r, d))
                                   if y_var and self.model.getSolVal(best_sol, y_var) is not None:
                                        var_value = self.model.getSolVal(best_sol, y_var)
                                        if var_value > 1e-4: # Usar una pequeña tolerancia
                                           assignment[(p, t, r, d)] = var_value
                 else:
                    print("DEBUG (Extract): model.getBestSol() returned None.")
                    status = "No Solution (No Best Sol)"


             except Exception as e:
                 print(f"Error extrayendo resultados SCIP: {e}")
                 status = "Error Extracting"
                 makespan_val = None # Reset values on extraction error
                 assignment = {}
                 task_completion_days = {}
                 objective_val = None
        else:
            print(f"DEBUG (Extract): Solver status was {status}. No solution found by getNSols().")
            status = f"No Solution ({status})" # Update status to reflect no solution found


        # self.solver_runtime should be set in the base class solve() method
        self.result = OptimizationResult(
            status=status,
            solver_runtime=self.solver_runtime,
            objective_value=objective_val,
            makespan=makespan_val,
            assignment=assignment,
            task_completion_days=task_completion_days
        )
        # Ensure objective_value and makespan are consistent if makespan is the objective
        # This check might be redundant if makespan_val is directly taken from objective_val,
        # but useful if makespan_val is derived or for robustness.
        if objective_val is not None and makespan_val is not None and abs(objective_val - makespan_val) < 1e-6:
             self.result.objective_value = makespan_val
        elif makespan_val is not None:
             # If makespan found (possibly derived), set it as objective_value if objective_val was None
             if self.result.objective_value is None:
                  self.result.objective_value = makespan_val