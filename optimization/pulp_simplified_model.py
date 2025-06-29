import pulp
import time
from .base_model import OptimizationModel
from .data_models import OptimizationResult, OptimizationInput

class SimplifiedPuLPModel(OptimizationModel):
    """
    Resuelve el problema de RCPSP simplificado (modelo de asignación diaria)
    usando un solver clásico MILP (PuLP) para obtener una solución óptima de referencia.
    """
    def __init__(self, input_data: OptimizationInput):
        """Constructor para inicializar atributos específicos del modelo."""
        super().__init__(input_data)
        # Definimos BLOCK_SIZE como un atributo de la clase
        self.BLOCK_SIZE = 8
    

    def _build_model(self):
        print("DEBUG PULP: Construyendo modelo MILP SIMPLIFICADO...")
        self.model = pulp.LpProblem("RCPSP_Simplified_PuLP", pulp.LpMinimize)

        # --- 1. CONSTANTES Y VARIABLES ---
        BLOCK_SIZE = 8
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        
        # z_rtd: 1 si recurso r trabaja en tarea t el día d
        z = pulp.LpVariable.dicts("Z", ((r, t, d) for r in self.resource_names for p in self.project_names for t in self.tasks_per_project_name.get(p,[]) for d in self.days_list), cat='Binary')
        
        # x_ptr: 1 si recurso r es asignado a la tarea (p,t)
        x = pulp.LpVariable.dicts("X", ((p, t, r) for p in self.project_names for t in self.tasks_per_project_name.get(p,[]) for r in self.resource_names), cat='Binary')

        # makespan: variable entera para el objetivo
        makespan = pulp.LpVariable("Makespan", lowBound=0, cat='Integer')

        # --- NUEVA VARIABLE AUXILIAR ---
        # WorkDay_td: 1 si la tarea t está activa el día d (trabajada por CUALQUIER recurso)
        WorkDay = pulp.LpVariable.dicts("WorkDay", ((t, d) for p in self.project_names for t in self.tasks_per_project_name.get(p,[]) for d in self.days_list), cat='Binary')
        
        self.variables = {'z': z, 'x': x, 'makespan': makespan, 'WorkDay': WorkDay}

        # --- 2. OBJETIVO ---
        self.model += makespan, "Minimizar_Makespan"

        # --- 3. RESTRICCIONES (VERSIÓN FINAL) ---

        # R1: Asignación Única
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                self.model += pulp.lpSum(x[(p, t, r)] for r in self.resource_names) == 1, f"AsignacionUnica_{p}_{t}"

        # R2: Expertise
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                req_level = level_map.get(self.expertise_required_dict.get((p, t), "Junior"), 1)
                for r in self.resource_names:
                    if level_map.get(self.expertise_dict.get(r, "Junior"), 1) < req_level:
                        self.model += x[(p, t, r)] == 0, f"Expertise_{p}_{t}_{r}"

        # R3: Vínculo Asignación-Trabajo (x -> z) - Versión más fuerte
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        self.model += z[(r, t, d)] <= x[(p, t, r)], f"Vinculo_xz_{p}_{t}_{r}_{d}"

        # R4: Horas Totales
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                H_pt = self.hours_required_dict.get((p, t), 0)
                self.model += pulp.lpSum(BLOCK_SIZE * z[(r, t, d)] for r in self.resource_names for d in self.days_list) >= H_pt, f"HorasTotales_{p}_{t}"
        
        # R5: Disponibilidad (VERSIÓN ROBUSTA)
        print("DEBUG PULP: Formulando P: Disponibilidad...")
        for r in self.resource_names:
            for d in self.days_list:
                if self.availability_numeric.get((r, d), 0) > 0:
                    # Si es un día laborable, solo puede hacer una tarea (un bloque de 8h)
                    self.model += pulp.lpSum(z[(r, t, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, [])) <= 1, f"Disponibilidad_{r}_{d}"
                else:
                    # Si NO es un día laborable, tiene PROHIBIDO trabajar
                    self.model += pulp.lpSum(z[(r, t, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, [])) == 0, f"NoWorkOnWeekend_{r}_{d}"

        # R6: Definición de Makespan
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        self.model += makespan >= d * z[(r, t, d)], f"DefMakespan_{r}_{t}_{d}"

        # R7: Vínculo z -> WorkDay (NUEVO)
        # Vincula nuestras variables principales 'z' con las auxiliares 'WorkDay'
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    # Si algún recurso trabaja (sum(z)>0), WorkDay debe ser 1.
                    self.model += WorkDay[(t,d)] <= pulp.lpSum(z[(r, t, d)] for r in self.resource_names), f"Link_z_Wd_upper_{t}_{d}"
                    # WorkDay solo puede ser 1 si algún recurso trabaja.
                    for r in self.resource_names:
                        self.model += WorkDay[(t,d)] >= z[(r, t, d)], f"Link_z_Wd_lower_{r}_{t}_{d}"

        # R8: Secuencialidad (NUEVO Y LÓGICAMENTE CORRECTO)
        # Usa las variables auxiliares 'WorkDay' para una formulación limpia.
        for p_proj in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name.get(p_proj, []),
                key=lambda t_name: getattr(self.task_info.get((p_proj, t_name)), 'sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i, t_j = sorted_tasks[i], sorted_tasks[i+1]
                
                # El último día de trabajo en t_i debe ser ANTES que el primer día de trabajo en t_j
                for d in self.days_list:
                    # El último día de trabajo en t_i es SUM(k*Wd_ik) / SUM(Wd_ik)
                    # El primer día de trabajo en t_j es SUM(d*Wd_jd) / SUM(Wd_jd)
                    # La restricción End(i) < Start(j) se puede modelar como:
                    # Para cada día 'd', si la tarea j está activa, la tarea i no puede estar activa en ningún día k>=d
                    for k in self.days_list:
                        if k >= d:
                            # No pueden estar activas a la vez si hay solapamiento temporal.
                            self.model += WorkDay[(t_i, k)] + WorkDay[(t_j, d)] <= 1, f"Secuencia_{t_i}_{t_j}_{d}_{k}"


    def _solve_model(self):
        start_time = time.time()
        # Usar el límite de tiempo de la configuración
        self.model.solve(pulp.PULP_CBC_CMD(timeLimit=self.input_data.config.solver_time_limit))
        runtime = time.time() - start_time
        status = pulp.LpStatus[self.model.status]
        return status, runtime

    def _extract_results(self, status):
        # Implementación simple para mostrar el resultado en consola
        if status in ["Optimal", "Feasible"]:
            makespan = self.variables['makespan'].varValue
            task_assignment = {}
            work_details = {}
            
            print("\n" + "="*20 + " INICIO SOLUCIÓN ÓPTIMA (PULP) " + "="*20)
            print(f"Makespan Óptimo Encontrado: {makespan:.0f} días")
            print("\n--- Asignaciones y Horas ---")
            
            for (p,t,r), v in self.variables['x'].items():
                if v.varValue > 0.9:
                    task_assignment[(p, t)] = r
                    
            for (r,t,d), v in self.variables['z'].items():
                if v.varValue > 0.9:
                    # Encontrar el proyecto 'p' para la tarea 't'
                    proj_name = [p_ for p_, t_list in self.tasks_per_project_name.items() if t in t_list][0]
                    work_details[(proj_name, t, r, d)] = self.BLOCK_SIZE
                    print(f"  - Día {d}: Tarea '{t}' -> Recurso: {r} ({self.BLOCK_SIZE}h)")

            print("="*22 + " FIN SOLUCIÓN ÓPTIMA (PULP) " + "="*23 + "\n")

            self.result = OptimizationResult(
                status="Optimal", 
                makespan=makespan, 
                solver_runtime=self.solver_runtime,
                assignment=task_assignment,
                work_details=work_details
            )
        else:
            self.result = OptimizationResult(status=status, solver_runtime=self.solver_runtime)