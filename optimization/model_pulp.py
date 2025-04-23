# optimization/makespan_model_pulp.py
import pulp
import time
from optimization.base_model import OptimizationModel
from optimization.data_models import OptimizationResult # Importar clase resultado
from datetime import date, timedelta

class MakespanMinimizationPuLP(OptimizationModel):
    """Minimiza Makespan usando PuLP (basado en v5.1)."""

    def _build_model(self):
        """Construye el modelo PuLP para minimizar makespan."""
        # Los datos ya están preparados en self.project_names, self.tasks_per_project_name, etc.
        # por _prepare_common_data() llamado en solve()

        self.model = pulp.LpProblem("Makespan_PuLP", pulp.LpMinimize)
        self.variables = {} # Reiniciar variables para este modelo

        # --- Definir Variables ---
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8
        # Asegurarse que self.M_days se calculó en _prepare_common_data
        M_days = getattr(self, 'M_days', len(getattr(self, 'days_list', [0])) + 1)

        # x, y, work_day, end_day, makespan
        x = pulp.LpVariable.dicts("assign", [(p, t, r) for p in self.project_names for t in self.tasks_per_project_name[p] for r in self.resource_names], cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts("hours", [(p, t, r, d) for p in self.project_names for t in self.tasks_per_project_name[p] for r in self.resource_names for d in self.days_list], lowBound=0, cat=pulp.LpContinuous)
        work_day = pulp.LpVariable.dicts("WorkDay", [(p, t, d) for p in self.project_names for t in self.tasks_per_project_name[p] for d in self.days_list], cat=pulp.LpBinary)
        end_day = pulp.LpVariable.dicts("EndDay", [(p, t) for p in self.project_names for t in self.tasks_per_project_name[p]], lowBound=1, cat=pulp.LpInteger)
        makespan = pulp.LpVariable("Makespan", lowBound=1, cat=pulp.LpInteger)

        # Guardar referencias
        self.variables = {'x': x, 'y': y, 'work_day': work_day, 'end_day': end_day, 'makespan': makespan}

        # --- Objetivo ---
        self.model += makespan, "Minimize_Makespan"

        # --- Restricciones ---
        # 4.1 Asignación única y validación de expertise
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                self.model += pulp.lpSum(x[(p, t, r)] for r in self.resource_names) == 1, f"assign_one_res_{p}_{t}"
                
                required_level = level_map.get(self.expertise_required_dict[(p, t)], 1)
                for r in self.resource_names:
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < required_level:
                        self.model += x[(p, t, r)] == 0, f"expertise_block_{p}_{t}_{r}"
                    
                    for d in self.days_list:
                        self.model += y[(p, t, r, d)] <= M_daily * x[(p, t, r)], f"link_y_x_{p}_{t}_{r}_{d}"

                # 4.1 Horas totales por tarea
                self.model += pulp.lpSum(y[(p, t, r, d)] for r in self.resource_names for d in self.days_list) == self.hours_required_dict[(p, t)], f"total_hours_{p}_{t}"

        # 4.2 Disponibilidad diaria de recursos
        for r in self.resource_names:
            for d in self.days_list:
                self.model += pulp.lpSum(y[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name[p]) <= self.availability_numeric.get((r, d), 0), f"avail_{r}_{d}"

        # 4.3 Enlace entre horas (y), días activos (work_day) y día de finalización (end_day)
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                for d in self.days_list:
                    for r in self.resource_names:
                        self.model += y[(p, t, r, d)] <= M_daily * work_day[(p, t, d)], f"link_y_work_{p}_{t}_{r}_{d}"
                    
                    self.model += end_day[(p, t)] >= d - M_days * (1 - work_day[(p, t, d)]), f"link_endday_work_{p}_{t}_{d}"

        # 4.4 Restricciones de secuencia entre tareas (Finish to Start)
        print("DEBUG (Build): Añadiendo restricciones de secuencia...")
        for p in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name[p],
                key=lambda t: self.task_info[(p, t)].sequence if (p, t) in self.task_info else float('inf')
            )
            for i in range(len(sorted_tasks) - 1):
                t_i = sorted_tasks[i]
                t_next = sorted_tasks[i+1]
                for d in self.days_list:
                    if d > 1:
                        self.model += end_day[(p, t_i)] <= (d - 1) + M_days * (1 - work_day[(p, t_next, d)]), f"seq_{p}_{t_i}_{t_next}_d{d}"
                    else:
                        if self.hours_required_dict[(p, t_i)] > 0:
                            self.model += work_day[(p, t_next, 1)] == 0, f"seq_d1_{p}_{t_i}_{t_next}"

        # 4.5 Restricciones de deadline (si existen)
        for project in self.input_data.projects:
            if project.deadline:
                deadline_day = (project.deadline - self.start_date).days + 1
                if deadline_day <= self.days_list[-1]:
                    for t in self.tasks_per_project_name[project.name]:
                        self.model += end_day[(project.name, t)] <= deadline_day, f"deadline_{project.name}_{t}"
                else:
                    print(f"⚠️ Deadline fuera del horizonte para proyecto {project.name}")

        # 4.6 Definición de Makespan
        for p in self.project_names:
            for t in self.tasks_per_project_name[p]:
                self.model += makespan >= end_day[(p, t)], f"makespan_def_{p}_{t}"
        

        print("DEBUG (Build): Modelo PuLP construido.")


    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el modelo PuLP construido."""
        if self.model is None:
            return "Error", 0.0
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=self.input_data.config.solver_time_limit)
        start_t = time.time()
        status_code = self.model.solve(solver)
        runtime = time.time() - start_t
        status_str = pulp.LpStatus[status_code]
        return status_str, runtime

    def _extract_results(self, status: str):
        """Extrae los resultados del modelo PuLP resuelto."""
        makespan_val = None
        assignment = {}
        task_completion_days = {}
        objective_val = None

        if status == "Optimal" or (status == "Not Solved" and self.model.objective is not None and pulp.value(self.model.objective) is not None):
            try:
                objective_val = pulp.value(self.model.objective)
                makespan_val = objective_val
                y = self.variables['y']
                end_day = self.variables['end_day']
                # Extract completion days
                for p in self.project_names:
                    for t in self.tasks_per_project_name[p]:
                        task_completion_days[(p, t)] = pulp.value(end_day[(p, t)])
                # Extract assignment
                for p in self.project_names:
                    for t in self.tasks_per_project_name[p]:
                        for r in self.resource_names:
                            for d in self.days_list:
                                var_value = pulp.value(y[(p, t, r, d)])
                                if var_value is not None and var_value > 1e-4:
                                    assignment[(p, t, r, d)] = var_value
            except Exception as e:
                print(f"Error extrayendo resultados PuLP: {e}")
                status = "Error Extracting"

        self.result = OptimizationResult(
            status=status,
            solver_runtime=self.solver_runtime,
            objective_value=objective_val,
            makespan=makespan_val,
            assignment=assignment,
            task_completion_days=task_completion_days
        )