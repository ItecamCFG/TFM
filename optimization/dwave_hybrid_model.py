# optimization/dwave_hybrid_model.py
import os
import time
import dimod
import math

# Importar el sampler híbrido de D-Wave
from dwave.system import LeapHybridSampler

from .base_model import OptimizationModel
from .data_models import OptimizationResult

# --- Funciones Auxiliares (sin cambios) ---

def integer_to_binary(label, max_value, prefix="int_"):
    if max_value < 0: max_value = 0
    num_bits = math.ceil(math.log2(max_value + 1)) if max_value > 0 else 1
    binary_vars_dict = {i: dimod.Binary(f"{prefix}{label}_b{i}") for i in range(num_bits)}
    linear_expression = dimod.quicksum((2**i) * binary_vars_dict[i] for i in range(num_bits))
    return linear_expression, binary_vars_dict

# --- Clase del Modelo ---

class DWaveHybridMakespanModel(OptimizationModel):
    """Minimiza Makespan usando D-Wave LeapHybridSampler."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = None
        self.sampleset = None
        self.variables = {}
        # El token se leerá de las variables de entorno, no se guarda en la instancia.

    def _build_model(self):
        """
        Construye el modelo QUBO (BQM).
        Esta función es IDÉNTICA a la de neal_model.py, ya que la formulación
        del problema no cambia, solo el solver que lo resuelve.
        """
        print("DEBUG HYBRID: Construyendo modelo BQM (idéntico a QUBO)...")
        if not hasattr(self, 'days_list') or not self.days_list:
            raise ValueError("days_list no disponible. Asegúrate que _prepare_common_data se ejecutó.")

        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8
        max_day_val = self.days_list[-1] if self.days_list else 1
        self.M_days_big_m = max_day_val + 10

        self.variables = {}
        all_model_binary_vars = []

        # 1. Variables Binarias (x, y, end_day, makespan, work_day)
        # (El código es el mismo que en neal_model.py)
        # x (Asignación)
        x_vars = {
            (p, t, r): dimod.Binary(f"x_{p}_{t}_{r}")
            for p in self.project_names
            for t in self.tasks_per_project_name.get(p, [])
            for r in self.resource_names
        }
        self.variables['x'] = x_vars

        # y (Horas)
        y_vars_bits, y_expressions = {}, {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        label = f"y_{p}_{t}_{r}_{d}"
                        expr, bits = integer_to_binary(label, M_daily, prefix="")
                        y_expressions[(p, t, r, d)] = expr
                        y_vars_bits[(p, t, r, d)] = bits
        self.variables['y_bits'] = y_vars_bits
        self.variables['y_expr'] = y_expressions

        # end_day
        end_day_vars_bits, end_day_expressions = {}, {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                label = f"EndDay_{p}_{t}"
                expr, bits = integer_to_binary(label, max_day_val, prefix="")
                end_day_expressions[(p, t)] = expr
                end_day_vars_bits[(p, t)] = bits
        self.variables['end_day_bits'] = end_day_vars_bits
        self.variables['end_day_expr'] = end_day_expressions

        # makespan
        makespan_expr, makespan_vars_bits = integer_to_binary("Makespan", max_day_val, prefix="")
        self.variables['makespan_bits'] = makespan_vars_bits
        self.variables['makespan_expr'] = makespan_expr

        # work_day
        work_day_vars = {
            (p, t, d): dimod.Binary(f"WorkDay_{p}_{t}_{d}")
            for p in self.project_names
            for t in self.tasks_per_project_name.get(p, [])
            for d in self.days_list
        }
        self.variables['work_day'] = work_day_vars

        # 2. Coeficientes de Penalización (Empíricos)
        P_BASE = max(1.0, float(max_day_val))
        P_HIGH = 10.0 * P_BASE**2
        P_MEDIUM = 2.0 * P_BASE
        P_LOW = 0.5 * P_BASE
        P_OBJECTIVE = 1.0
        
        P_LINK = 5.0 * P_BASE
        P_HOURS_TOTAL = 8.0 * P_BASE
        P_AVAIL = 8.0 * P_BASE
        P_SEQ = 10.0 * P_BASE
        P_DEADLINE = 15.0 * P_BASE
        P_MAKE_DEF = 10.0 * P_BASE

        # 3. Construcción del BQM
        bqm = dimod.BinaryQuadraticModel('BINARY')
        bqm.update(P_OBJECTIVE * makespan_expr)

        # 4. Restricciones (idénticas a neal_model.py, incluidas todas hasta la 10)
        print("DEBUG HYBRID: Formulando todas las restricciones (1-10)...")

        # 1. Asignación única
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                sum_x = dimod.quicksum(x_vars[(p, t, r)] for r in self.resource_names)
                bqm.update(P_HIGH * (sum_x - 1)**2)

        # 2. Expertise
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                req_level = level_map.get(self.expertise_required_dict.get((p, t), "Junior"), 1)
                for r in self.resource_names:
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < req_level:
                        bqm.update(P_HIGH * x_vars[(p, t, r)])

        # 3. Link y-x
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_var = x_vars[(p, t, r)]
                    for d in self.days_list:
                        for bit_y_var in y_vars_bits[(p, t, r, d)].values():
                            bqm.update(P_LINK * bit_y_var * (1 - x_var))
        
        # 4. Horas Totales
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                H_pt = self.hours_required_dict.get((p, t), 0)
                sum_y_expr_pt = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names for d in self.days_list)
                bqm.update(P_HOURS_TOTAL * (sum_y_expr_pt - H_pt)**2)

        # 5. Disponibilidad
        for r in self.resource_names:
            for d in self.days_list:
                A_rd = self.availability_numeric.get((r, d), 0)
                sum_y_rd = dimod.quicksum(y_expressions[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, []))
                slack_expr, _ = integer_to_binary(f"SlackAvail_{r}_{d}", A_rd, prefix="")
                bqm.update(P_AVAIL * (A_rd - sum_y_rd - slack_expr)**2)

        # 6. Enlace y -> WorkDay
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    work_var = work_day_vars[(p,t,d)]
                    sum_y_ptd = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names)
                    # Si sum_y > 0, work_day debe ser 1. y_sum - M*work_day <= 0
                    bqm.add_linear_inequality_constraint(
                        terms=[(bit, coef) for r in self.resource_names for bit, coef in y_expressions[(p, t, r, d)].linear.items()] + [(work_var, -M_daily)],
                        lagrange_multiplier=P_LINK,
                        label=f"link_y_sum_workday_{p}_{t}_{d}",
                        ub=0
                    )

        # 7. Definición EndDay
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                end_expr_pt = end_day_expressions[(p,t)]
                for d_loop in self.days_list:
                    work_var_ptd = work_day_vars[(p,t,d_loop)]
                    # end_expr - d*work_day >= 0 (aproximado, mejor con Big-M)
                    # end_expr >= d - M_days*(1-work_day) => end_expr + M*work_day >= d
                    bqm.add_linear_inequality_constraint(
                        terms=[(bit, coef) for bit, coef in end_expr_pt.linear.items()] + [(work_var_ptd, self.M_days_big_m)],
                        lagrange_multiplier=P_HIGH,
                        label=f"EndDay_def_{p}_{t}_{d_loop}",
                        lb=d_loop
                    )

        # 8. Secuencialidad
        for p_proj in self.project_names:
            sorted_tasks = sorted(self.tasks_per_project_name[p_proj], key=lambda t_name: self.task_info.get((p_proj, t_name), {}).get('sequence', float('inf')))
            for i in range(len(sorted_tasks) - 1):
                t_i, t_i_plus_1 = sorted_tasks[i], sorted_tasks[i+1]
                end_i_expr = end_day_expressions[(p_proj, t_i)]
                for d_val in self.days_list:
                    if d_val > 1:
                        work_next = work_day_vars[(p_proj, t_i_plus_1, d_val)]
                        # end_i <= d-1 + M*(1-work_next) => end_i + M*work_next <= d-1+M
                        bqm.add_linear_inequality_constraint(
                            terms=[(b, c) for b, c in end_i_expr.linear.items()] + [(work_next, self.M_days_big_m)],
                            lagrange_multiplier=P_SEQ,
                            label=f"Seq_{p_proj}_{t_i}_{d_val}",
                            ub=(d_val - 1 + self.M_days_big_m)
                        )
        
        # 9. Deadline
        for project_data in self.input_data.projects:
            p_name = project_data.name
            if project_data.deadline:
                deadline_day_num = (project_data.deadline - self.start_date).days + 1
                if 1 <= deadline_day_num <= max_day_val:
                    for t_task in self.tasks_per_project_name.get(p_name,[]):
                        end_expr_pt = end_day_expressions[(p_name, t_task)]
                        bqm.add_linear_inequality_constraint(
                           terms=[(b,c) for b,c in end_expr_pt.linear.items()],
                           lagrange_multiplier=P_DEADLINE,
                           label=f"Deadline_{p_name}_{t_task}",
                           ub=deadline_day_num
                        )

        # 10. Definición Makespan
        for p_proj in self.project_names:
            for t_task in self.tasks_per_project_name.get(p_proj, []):
                end_expr_pt = end_day_expressions[(p_proj, t_task)]
                # makespan >= end_day => makespan - end_day >= 0
                bqm.add_linear_inequality_constraint(
                    terms=[(b, c) for b, c in makespan_expr.linear.items()] + [(b, -c) for b, c in end_expr_pt.linear.items()],
                    lagrange_multiplier=P_MAKE_DEF,
                    label=f"MakespanDef_{p_proj}_{t_task}",
                    lb=0
                )

        self.model = bqm
        num_vars_final = len(self.model.variables)
        print(f"DEBUG HYBRID: Modelo BQM final construido con {num_vars_final} variables binarias.")


    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el BQM usando D-Wave LeapHybridSampler."""
        if self.model is None:
            return "Error Building", 0.0

        # --- CAMBIO PRINCIPAL: Usar LeapHybridSampler ---
        # El token se lee automáticamente de la variable de entorno DWAVE_API_TOKEN
        # o del fichero de configuración de D-Wave si existe.
        try:
            sampler = LeapHybridSampler()
        except Exception as e:
            error_msg = f"Error al inicializar el sampler de D-Wave: {e}. "
            error_msg += "Asegúrate de que tu API token está configurado como variable de entorno (DWAVE_API_TOKEN)."
            print(error_msg)
            return "API Token Error", 0.0
        
        print("DEBUG HYBRID: Enviando problema al LeapHybridSampler...")
        start_t = time.time()
        
        # El tiempo límite se pasa como parámetro al método sample()
        time_limit = self.input_data.config.solver_time_limit
        
        try:
            self.sampleset = sampler.sample(self.model, time_limit=time_limit, label=f"TFM-ProjectPlanning-{int(time.time())}")
        except Exception as e:
            print(f"ERROR HYBRID: Falló la ejecución del LeapHybridSampler: {e}")
            import traceback
            print(traceback.format_exc())
            self.sampleset = None
            return "Error Sampling", time.time() - start_t
            
        runtime = time.time() - start_t
        print(f"DEBUG HYBRID: LeapHybridSampler finalizado en {runtime:.2f} segundos.")

        if self.sampleset and len(self.sampleset) > 0:
            return "Feasible", runtime
        else:
            return "No Solution Found", runtime

    def _extract_results(self, status: str):
        """
        Extrae los resultados del sampleset.
        Esta función es IDÉNTICA a la de neal_model.py.
        """
        makespan_val, assignment, task_completion_days, objective_val = None, {}, {}, None
        best_sample, error_message, final_status = None, None, status

        if status in ["Feasible", "No Solution Found"] and hasattr(self, 'sampleset') and self.sampleset is not None and len(self.sampleset) > 0:
            try:
                best_sample = self.sampleset.first.sample
                objective_val = self.sampleset.first.energy
                print(f"DEBUG HYBRID: Mejor energía encontrada: {objective_val}")

                # Decodificar Makespan
                makespan_val = 0
                if 'makespan_bits' in self.variables:
                    for i, bit_var in self.variables['makespan_bits'].items():
                        if best_sample.get(bit_var.label, 0) == 1: makespan_val += 2**i

                # Decodificar EndDay
                if 'end_day_bits' in self.variables:
                    for (p, t), bits_dict in self.variables['end_day_bits'].items():
                        day_val = sum(2**i for i, bit_var in bits_dict.items() if best_sample.get(bit_var.label, 0) == 1)
                        task_completion_days[(p, t)] = day_val
                
                # Decodificar y (Horas)
                if 'y_bits' in self.variables:
                    for (p,t,r,d), bits_dict in self.variables['y_bits'].items():
                        hours_val = sum(2**i for i, bit_var in bits_dict.items() if best_sample.get(bit_var.label, 0) == 1)
                        if hours_val > 1e-4:
                            assignment[(p,t,r,d)] = hours_val

                # Validación (Opcional pero recomendado)
                # ...
                final_status = "Feasible (Validation Pending)"

            except Exception as e:
                import traceback
                error_message = f"Error extrayendo/decodificando: {e}\n{traceback.format_exc()}"
                final_status = "Error Extracting"
        
        # Calcular coste total (opcional)
        total_cost = sum(self.cost_dict.get(r,0.0) * h for (_,_,r,_),h in assignment.items()) if assignment else None

        self.result = OptimizationResult(
            status=final_status, solver_runtime=self.solver_runtime,
            objective_value=objective_val, makespan=makespan_val, total_cost=total_cost,
            assignment=assignment, task_completion_days=task_completion_days, error_message=error_message
        )