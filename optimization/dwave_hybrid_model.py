# optimization/dwave_hybrid_model.py
import os
import time
import dimod
import math

# Importar el sampler híbrido de D-Wave
from dwave.system import LeapHybridSampler

from .base_model import OptimizationModel
from .data_models import OptimizationResult

# --- Función Auxiliar Corregida ---
def integer_to_binary(label, max_value, prefix="int_"):
    """
    Devuelve una expresión lineal (BQM) y un diccionario con los NOMBRES (labels)
    de las variables binarias que la componen.
    """
    if max_value <= 0: max_value = 1
    num_bits = math.ceil(math.log2(max_value + 1)) if max_value > 0 else 1
    
    # Crear y guardar los nombres (strings) de las variables
    binary_var_labels = {i: f"{prefix}{label}_b{i}" for i in range(num_bits)}
    
    # Crear la expresión lineal usando los nombres de las variables
    linear_expression = dimod.quicksum(
        (2**i) * dimod.Binary(var_label) for i, var_label in binary_var_labels.items()
    )
    
    return linear_expression, binary_var_labels

# --- Clase del Modelo ---
class DWaveHybridMakespanModel(OptimizationModel):
    """Minimiza Makespan usando D-Wave LeapHybridSampler."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = None
        self.sampleset = None
        self.variables = {}

    def _build_model(self):
        """
        Construye el modelo QUBO (BQM) con lógica de restricciones corregida y refactorizada.
        """
        print("DEBUG HYBRID: Construyendo el modelo BQM con lógica de restricciones corregida...")
        if not hasattr(self, 'days_list') or not self.days_list:
            raise ValueError("Datos no preparados.")

        # --- 1. Parámetros y Variables (sin cambios) ---
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        max_day_val = self.days_list[-1] if self.days_list else 1
        self.variables = {}

        x_vars = {(p, t, r): f"x_{p}_{t}_{r}" for p in self.project_names for t in self.tasks_per_project_name.get(p, []) for r in self.resource_names}
        self.variables['x'] = x_vars

        y_var_labels, y_expressions = {}, {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        expr, labels = integer_to_binary(f"y_{p}_{t}_{r}_{d}", 8)
                        y_expressions[(p, t, r, d)] = expr
                        y_var_labels[(p, t, r, d)] = labels
        self.variables['y_bits'] = y_var_labels
        self.variables['y_expr'] = y_expressions

        end_day_labels, end_day_expressions = {}, {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                expr, labels = integer_to_binary(f"EndDay_{p}_{t}", max_day_val)
                end_day_expressions[(p, t)] = expr
                end_day_labels[(p, t)] = labels
        self.variables['end_day_bits'] = end_day_labels

        makespan_expr, makespan_labels = integer_to_binary("Makespan", max_day_val)
        self.variables['makespan_bits'] = makespan_labels

        # --- 2. Penalizaciones ---
        P_BASE = max(1.0, float(max_day_val))
        P_HIGH = 15.0 * P_BASE**2  # Aumentada para restricciones duras
        P_MEDIUM = 5.0 * P_BASE
        P_OBJECTIVE = 1.0

        # --- 3. BQM y Objetivo ---
        bqm = dimod.BinaryQuadraticModel('BINARY')
        bqm.update(P_OBJECTIVE * makespan_expr)

        # --- 4. Aplicación de Restricciones ---
        print("DEBUG HYBRID: Aplicando restricciones refactorizadas...")

        # A. RESTRICCIONES POR TAREA
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                
                # A.1. Asignación única de recurso a la tarea
                sum_x = dimod.quicksum(dimod.Binary(x_vars[(p, t, r)]) for r in self.resource_names)
                bqm.update(P_HIGH * (sum_x - 1)**2)

                # A.2. Horas totales para la tarea
                H_pt = self.hours_required_dict.get((p, t), 0)
                sum_y_expr = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names for d in self.days_list)
                bqm.update(P_HIGH * (sum_y_expr - H_pt)**2)

                # A.3. Definición del día de finalización de la tarea
                for d in self.days_list:
                    sum_y_ptd = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names)
                    penalty_expr = (d - end_day_expressions[(p, t)]) * sum_y_ptd
                    bqm.update(P_HIGH * penalty_expr)

                # A.4. El Makespan global debe ser mayor o igual al día de finalización de esta tarea
                slack_makespan, _ = integer_to_binary(f"SlackMakespan_{p}_{t}", max_day_val)
                bqm.update(P_HIGH * (makespan_expr - slack_makespan - end_day_expressions[(p,t)])**2)

        # B. RESTRICCIONES POR RECURSO
        for r in self.resource_names:
            # B.1. Disponibilidad diaria del recurso
            for d in self.days_list:
                A_rd = self.availability_numeric.get((r, d), 0)
                sum_y_rd = dimod.quicksum(y_expressions[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, []))
                slack_avail, _ = integer_to_binary(f"SlackAvail_{r}_{d}", A_rd)
                bqm.update(P_HIGH * (sum_y_rd + slack_avail - A_rd)**2)
        
        # C. RESTRICCIONES DE ASIGNACIÓN (Combinan Tarea y Recurso)
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_var = dimod.Binary(x_vars[(p,t,r)])

                    # C.1. Expertise del recurso
                    req_level = level_map.get(self.expertise_required_dict.get((p, t), "Junior"), 1)
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < req_level:
                        bqm.update(P_HIGH * x_var)

                    # C.2. Enlace: solo se trabajan horas si está asignado
                    for d in self.days_list:
                        y_sum_bits = dimod.quicksum(dimod.Binary(label) for label in y_var_labels[(p,t,r,d)].values())
                        bqm.update(P_MEDIUM * y_sum_bits * (1 - x_var))
        
        # D. RESTRICCIONES DE SECUENCIA Y DEADLINE (Nivel Proyecto)
        for p in self.project_names:
            # D.1. Secuenciación de tareas dentro del proyecto (LÓGICA CORREGIDA)
            sorted_tasks = sorted(
                self.tasks_per_project_name[p], 
                key=lambda t_name: self.task_info.get((p, t_name)).sequence if self.task_info.get((p, t_name)) else float('inf')
            )
            for i in range(len(sorted_tasks) - 1):
                t_i, t_i_plus_1 = sorted_tasks[i], sorted_tasks[i+1]
                end_i = end_day_expressions[(p, t_i)]
                end_i_plus_1 = end_day_expressions[(p, t_i_plus_1)]
                
                # Nueva Lógica: end_day(i) <= end_day(i+1).
                # Se modela como end_day(i) - end_day(i+1) + slack = 0, donde slack >= 0.
                slack_seq, _ = integer_to_binary(f"SlackSeq_{p}_{t_i}", max_day_val)
                bqm.update(P_HIGH * (end_i - end_i_plus_1 + slack_seq)**2)

            # D.2. Deadline del proyecto
            project_obj = next((proj for proj in self.input_data.projects if proj.name == p), None)
            if project_obj and project_obj.deadline:
                deadline_day = (project_obj.deadline - self.start_date).days + 1
                if 1 <= deadline_day <= max_day_val:
                    # El deadline aplica a la última tarea de la secuencia
                    if sorted_tasks:
                        last_task_name = sorted_tasks[-1]
                        end_expr_last_task = end_day_expressions[(p, last_task_name)]
                        slack_deadline, _ = integer_to_binary(f"SlackDeadline_{p}", max_day_val)
                        bqm.update(P_HIGH * (end_expr_last_task + slack_deadline - deadline_day)**2)

        self.model = bqm
        num_vars_final = len(self.model.variables)
        print(f"DEBUG HYBRID: Modelo BQM final construido con {num_vars_final} variables binarias.")


    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el BQM usando D-Wave LeapHybridSampler."""
        if self.model is None: return "Error Building", 0.0
        try:
            token = os.getenv("DWAVE_API_TOKEN")
            if not token:
                raise ValueError("La variable de entorno DWAVE_API_TOKEN no está configurada.")
            sampler = LeapHybridSampler(token=token)
        except Exception as e:
            print(f"Error al inicializar D-Wave Sampler: {e}")
            return "API Token Error", 0.0
        
        print("DEBUG HYBRID: Enviando problema al LeapHybridSampler...")
        start_t = time.time()
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
        qpu_time_sec = self.sampleset.info.get('qpu_access_time', 0) / 1_000_000
        print(f"DEBUG HYBRID: Sampler finalizado en {runtime:.2f}s (tiempo QPU: {qpu_time_sec:.4f}s).")

        return ("Feasible", runtime) if self.sampleset and len(self.sampleset) > 0 else ("No Solution", runtime)

    def _extract_results(self, status: str):
        """
        Extrae los resultados del sampleset de forma robusta y con depuración.
        """
        if not (status == "Feasible" and hasattr(self, 'sampleset') and self.sampleset):
            self.result = OptimizationResult(status=status, solver_runtime=self.solver_runtime)
            return

        best_sample = self.sampleset.first.sample
        objective_val = self.sampleset.first.energy
        print(f"DEBUG HYBRID: Mejor energía encontrada: {objective_val:.2f}")

        makespan_val, assignment, task_completion_days, total_cost = None, {}, {}, None
        error_message = None
        final_status = status

        try:
            # === DECODIFICACIÓN CORREGIDA ===
            print("DEBUG EXTRACT: Decodificando resultados...")
            
            # Usamos los NOMBRES (labels) de las variables para buscar en la solución
            makespan_val = sum(2**i for i, label in self.variables['makespan_bits'].items() if best_sample.get(label, 0) == 1)

            task_completion_days = {
                (p, t): sum(2**i for i, label in labels.items() if best_sample.get(label, 0) == 1)
                for (p, t), labels in self.variables['end_day_bits'].items()
            }
            
            for (p, t, r, d), labels in self.variables['y_bits'].items():
                hours = sum(2**i for i, label in labels.items() if best_sample.get(label, 0) == 1)
                if hours > 1e-4:
                    assignment[(p, t, r, d)] = hours
            
            if assignment and hasattr(self, 'cost_dict'):
                total_cost = sum(self.cost_dict.get(r, 0.0) * h for (_, _, r, _), h in assignment.items())

            final_status = "Feasible (Validation Pending)"
            print("DEBUG EXTRACT: Decodificación completada con éxito.")

        except Exception as e:
            import traceback
            error_message = f"Error durante la decodificación de resultados: {e}\n{traceback.format_exc()}"
            print(f"ERROR EXTRACT: {error_message}")
            final_status = "Error Extracting"

        self.result = OptimizationResult(
            status=final_status,
            solver_runtime=self.solver_runtime,
            objective_value=objective_val,
            makespan=makespan_val,
            total_cost=total_cost,
            assignment=assignment,
            task_completion_days=task_completion_days,
            error_message=error_message
        )