# optimization/neal_model.py
import neal # pip install dwave-neal
import dimod # pip install dimod
import numpy as np
import math
import time
from datetime import date, timedelta

# Asegúrate que las rutas sean correctas según tu estructura
from .base_model import OptimizationModel
from data_models import OptimizationResult

# Helper para codificación binaria de enteros (igual que antes)
def integer_to_binary(label, max_value, prefix="int_"):
    """Crea variables binarias para representar un entero >= 0."""
    if max_value < 0: max_value = 0 # Asegurar no negativo
    num_bits = math.ceil(math.log2(max_value + 1)) if max_value > 0 else 1
    binary_vars = {i: dimod.Binary(f"{prefix}{label}_b{i}") for i in range(num_bits)}
    linear_expression = dimod.quicksum( (2**i) * binary_vars[i] for i in range(num_bits) )
    return linear_expression, binary_vars

class NealMakespanModel(OptimizationModel):
    """Minimiza Makespan usando Neal Simulated Annealing (QUBO)."""

    def _build_model(self):
        """Construye el modelo QUBO (BQM) para minimizar makespan."""
        print("DEBUG QUBO: Construyendo modelo BQM...")
        # --- Preparación Inicial ---
        # Asumimos que _prepare_common_data ya se llamó y pobló:
        # self.project_names, self.tasks_per_project_name, self.resource_names,
        # self.expertise_dict, self.cost_dict, self.hours_required_dict,
        # self.expertise_required_dict, self.task_info, self.days_list,
        # self.availability_numeric, self.start_date
        if not hasattr(self, 'days_list') or not self.days_list:
             raise ValueError("La lista de días (days_list) no está disponible o está vacía.")

        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8 # Max horas por tarea/recurso/día
        max_day = self.days_list[-1]
        self.M_days = max_day + 1 # BigM para días

        self.variables = {} # Reiniciar
        all_model_variables = [] # Para añadir al BQM final

        # --- 1. Variables Binarias ---

        # x (Asignación)
        x_vars = {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    key = f"x_{p}_{t}_{r}"
                    x_vars[(p, t, r)] = dimod.Binary(key)
                    all_model_variables.append(x_vars[(p, t, r)])
        self.variables['x'] = x_vars
        print(f"DEBUG QUBO: {len(x_vars)} variables 'x'.")

        # y (Horas) - Codificación binaria
        y_vars_bits = {}
        y_expressions = {}
        num_bits_y = math.ceil(math.log2(M_daily + 1)) if M_daily > 0 else 1
        count_y_vars = 0
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        label = f"y_{p}_{t}_{r}_{d}"
                        expr, bits = integer_to_binary(label, M_daily, prefix="")
                        y_expressions[(p, t, r, d)] = expr
                        y_vars_bits[(p, t, r, d)] = bits
                        all_model_variables.extend(bits.values())
                        count_y_vars += num_bits_y
        self.variables['y_bits'] = y_vars_bits
        self.variables['y_expr'] = y_expressions
        print(f"DEBUG QUBO: {count_y_vars} variables binarias para 'y'.")

        # end_day - Codificación binaria
        end_day_vars_bits = {}
        end_day_expressions = {}
        num_bits_end = math.ceil(math.log2(max_day + 1)) if max_day > 0 else 1
        count_end_vars = 0
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                label = f"EndDay_{p}_{t}"
                expr, bits = integer_to_binary(label, max_day, prefix="")
                end_day_expressions[(p, t)] = expr
                end_day_vars_bits[(p, t)] = bits
                all_model_variables.extend(bits.values())
                count_end_vars += num_bits_end
        self.variables['end_day_bits'] = end_day_vars_bits
        self.variables['end_day_expr'] = end_day_expressions
        print(f"DEBUG QUBO: {count_end_vars} variables binarias para 'end_day'.")

        # makespan - Codificación binaria
        makespan_expr, makespan_vars_bits = integer_to_binary("Makespan", max_day, prefix="")
        self.variables['makespan_bits'] = makespan_vars_bits
        self.variables['makespan_expr'] = makespan_expr
        all_model_variables.extend(makespan_vars_bits.values())
        print(f"DEBUG QUBO: {num_bits_end} variables binarias para 'makespan'.")

        # --- Coeficientes de Penalización (¡AJUSTAR!) ---
        P_BASE = max(1.0, float(max_day)) # Escala base
        P_ASSIGN = 10 * P_BASE**2 # Asignación, expertise (muy importantes)
        P_HOURS_TOTAL = 1.5 * P_BASE # Horas totales por tarea
        P_AVAIL = 2.0 * P_BASE # Disponibilidad diaria (si se implementa)
        P_LINK = 0.5 * P_BASE # Enlaces y-x
        P_SEQ = 5.0 * P_BASE**2 # Secuencia (importante)
        P_DEADLINE = 10.0 * P_BASE**2 # Deadline (muy importante)
        P_MAKE_DEF = 1.0 * P_BASE # Definición makespan (si se usa penalidad)
        P_OBJECTIVE = 1.0 # Peso del makespan en el objetivo final

        # --- BQM Principal ---
        bqm = dimod.BinaryQuadraticModel('BINARY')

        # --- Objetivo QUBO: Minimizar Makespan ---
        bqm.update(P_OBJECTIVE * makespan_expr)

        # --- Restricciones como Penalizaciones ---

        # 1. Asignación única: (sum(x) - 1)^2
        print("DEBUG QUBO: Formulando penalización asignación única...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                sum_x = dimod.quicksum(x_vars[(p, t, r)] for r in self.resource_names)
                bqm.update(P_ASSIGN * (sum_x - 1)**2)

        # 2. Expertise: P * x si no cumple
        print("DEBUG QUBO: Formulando penalización expertise...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                task_key = (p, t)
                req_level = level_map.get(self.expertise_required_dict.get(task_key, "Junior"), 1)
                for r in self.resource_names:
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < req_level:
                        bqm.update(P_ASSIGN * x_vars[(p, t, r)])

        # 3. Link y-x: y <= M*x. Penalizar si y > 0 y x = 0.
        # Usaremos la penalización por bit: P * y_bit * (1 - x)
        print("DEBUG QUBO: Formulando penalización link y-x...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_var = x_vars[(p, t, r)]
                    for d in self.days_list:
                        # Penalizar cada bit de y si x es 0
                        for bit_var in y_vars_bits[(p, t, r, d)].values():
                             bqm.update(P_LINK * bit_var * (1 - x_var))

        # 4. Horas Totales: (sum(y_expr) - H_pt)^2
        print("DEBUG QUBO: Formulando penalización horas totales...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                task_key = (p, t)
                H_pt = self.hours_required_dict.get(task_key, 0)
                sum_y_expr_pt = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names for d in self.days_list)
                bqm.update(P_HOURS_TOTAL * (sum_y_expr_pt - H_pt)**2)

        # 5. Disponibilidad: sum(y[p,t,r,d] for p,t) <= A_rd
        # Penalización: P_AVAIL * (sum(y_expr) + slack_avail - A_rd)^2
        print("DEBUG QUBO: Formulando penalización disponibilidad (con slack)...")
        availability_slacks = {} # Guardar { (r,d): (expr, bits) }
        num_bits_avail_slack = math.ceil(math.log2(len(self.project_names)*len(self.tasks_per_project_name.get(p,[]))*M_daily+1)) # Max horas posibles + 1
        for r in self.resource_names:
             for d in self.days_list:
                 A_rd = self.availability_numeric.get((r, d), 0)
                 sum_y_rd = dimod.quicksum(y_expressions[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, []))

                 # Crear slack binario
                 slack_label = f"SlackAvail_{r}_{d}"
                 slack_expr, slack_bits = integer_to_binary(slack_label, A_rd) # Slack max = A_rd
                 availability_slacks[(r, d)] = {'expr': slack_expr, 'bits': slack_bits}
                 all_model_variables.extend(slack_bits.values())

                 # Penalizar la igualdad: sum(y) + slack - A_rd == 0
                 bqm.update(P_AVAIL * (sum_y_rd + slack_expr - A_rd)**2)

        # 6 & 7. Enlace EndDay y Trabajo: EndDay >= d si se trabaja.
        # Implementado como EndDay >= d - M*(1-w). Requiere w.
        # work_day (z) no se usa explícitamente aquí aún.
        # La definición implícita de EndDay a través de las otras restricciones y el objetivo
        # puede ser suficiente, pero añadir penalizaciones explícitas puede ayudar.
        
        print("WARN QUBO: Enlace explícito EndDay-Trabajo NO implementado (confiando en otras restricciones).")

        # 8. Secuencialidad: EndDay[i] <= EndDay[i+1] (aproximado)
        # Penalización: P_SEQ * (EndDay[i] - EndDay[i+1] + SlackSeq)^2 = 0
        print("DEBUG QUBO: Formulando penalización secuencia (EndDay[i] <= EndDay[i+1] con slack)...")
        sequence_slacks = {}
        num_bits_seq_slack = num_bits_end # Slack max = max_day
        for p in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name[p],
                key=lambda t: self.task_info.get((p, t), {}).get('sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i = sorted_tasks[i]
                t_i_plus_1 = sorted_tasks[i+1]
                end_i_expr = end_day_expressions[(p, t_i)]
                end_i_plus_1_expr = end_day_expressions[(p, t_i_plus_1)]

                # Crear slack binario
                slack_label = f"SlackSeq_{p}_{t_i}_{t_i_plus_1}"
                slack_expr, slack_bits = integer_to_binary(slack_label, max_day)
                sequence_slacks[(p, i)] = {'expr': slack_expr, 'bits': slack_bits}
                all_model_variables.extend(slack_bits.values())

                # Penalizar igualdad: end_i - end_{i+1} + slack = 0
                bqm.update(P_SEQ * (end_i_expr - end_i_plus_1_expr + slack_expr)**2)

        # 9. Deadline: EndDay[p,t] <= D_p
        # Penalización: P_DEADLINE * (EndDay[p,t] + SlackDL - D_p)^2 = 0
        print("DEBUG QUBO: Formulando penalización deadline (con slack)...")
        deadline_slacks = {}
        num_bits_dl_slack = num_bits_end # Slack max = max_day
        for project_data in self.input_data.projects:
            p_name = project_data.name
            deadline_date = project_data.deadline
            if deadline_date:
                deadline_day_number = (deadline_date - self.start_date).days + 1
                if 1 <= deadline_day_number <= max_day: # Solo si deadline es alcanzable
                     for t in self.tasks_per_project_name.get(p_name,[]):
                          end_expr = end_day_expressions[(p_name, t)]

                          # Crear slack binario
                          slack_label = f"SlackDL_{p_name}_{t}"
                          # Slack máximo necesario es max_day - 1 (si deadline=1, end=1 -> slack=0; si end=max_day, deadline=1 -> slack=max_day-1)
                          # Pero para la fórmula End + Slack = Deadline, slack max es Deadline
                          slack_expr, slack_bits = integer_to_binary(slack_label, deadline_day_number)
                          deadline_slacks[(p_name, t)] = {'expr': slack_expr, 'bits': slack_bits}
                          all_model_variables.extend(slack_bits.values())

                          # Penalizar igualdad: end_day + slack - deadline = 0
                          bqm.update(P_DEADLINE * (end_expr + slack_expr - deadline_day_number)**2)

        # 10. Makespan Definition: Makespan >= EndDay[p,t]
        # Penalización: P_MAKE_DEF * (Makespan - EndDay[p,t] + SlackMK)^2 = 0
        print("DEBUG QUBO: Formulando penalización definición Makespan (con slack)...")
        makespan_slacks = {}
        num_bits_mk_slack = num_bits_end # Slack max = max_day
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                end_expr = end_day_expressions[(p, t)]

                # Crear slack binario
                slack_label = f"SlackMK_{p}_{t}"
                slack_expr, slack_bits = integer_to_binary(slack_label, max_day)
                makespan_slacks[(p, t)] = {'expr': slack_expr, 'bits': slack_bits}
                all_model_variables.extend(slack_bits.values())

                # Penalizar igualdad: makespan - end_day + slack = 0
                bqm.update(P_MAKE_DEF * (makespan_expr - end_expr + slack_expr)**2)


        # --- Finalizar BQM ---
        # Añadir todas las variables binarias que no tengan ya sesgo/interacción
        for var in all_model_variables:
             var_name = var.variables[0] # Nombre de la variable binaria
             if var_name not in bqm.variables:
                  bqm.add_variable(var_name, 0.0)

        self.model = bqm # Guardar el BQM construido
        num_vars_final = len(self.model.variables)
        print(f"DEBUG QUBO: Modelo BQM final construido con {num_vars_final} variables binarias.")
        if num_vars_final > 5000: # Advertir si es muy grande
             print(f"ADVERTENCIA QUBO: El número de variables ({num_vars_final}) es muy grande, Neal puede tardar mucho o fallar.")
        # print("DEBUG QUBO: BQM Lineal:", self.model.linear)
        # print("DEBUG QUBO: BQM Cuadrático:", self.model.quadratic)

    # --- _solve_model y _extract_results (Sin cambios respecto a la versión anterior) ---
    # ... (El código para llamar a neal y decodificar/validar resultados permanece igual) ...
    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el modelo QUBO usando Neal."""
        if self.model is None or not isinstance(self.model, dimod.BinaryQuadraticModel):
            print("ERROR QUBO: Modelo BQM no construido correctamente.")
            return "Error Building", 0.0

        sampler = neal.SimulatedAnnealingSampler()
        # Usar el time limit para calcular num_reads? O fijar num_reads?
        # Neal no tiene un time_limit directo. num_reads es el parámetro principal.
        # Ajustar num_reads basado en complejidad y tiempo deseado
        num_vars = len(self.model.variables)
        if num_vars < 100: num_reads = 1000
        elif num_vars < 1000: num_reads = 500
        else: num_reads = 100 # Reducir para modelos grandes

        print(f"DEBUG QUBO: Iniciando Neal Sampler con {num_reads} reads...")
        start_t = time.time()
        try:
            self.sampleset = sampler.sample(self.model, num_reads=num_reads)
        except Exception as e:
            print(f"ERROR QUBO: Falló la ejecución de Neal: {e}")
            import traceback
            print(traceback.format_exc())
            self.sampleset = None
            return "Error Sampling", time.time() - start_t

        runtime = time.time() - start_t
        print(f"DEBUG QUBO: Neal Sampler finalizado en {runtime:.2f} segundos.")

        if self.sampleset and len(self.sampleset) > 0:
            return "Feasible", runtime # Neal no garantiza optimalidad ni factibilidad estricta
        else:
            return "No Solution Found", runtime


    def _extract_results(self, status: str):
        """Extrae e interpreta los resultados del sampleset de Neal."""
        # --- (Mismo código de extracción/decodificación que antes) ---
        # ... (incluyendo decodificación de makespan, end_day, y) ...
        # ... (incluyendo cálculo de total_cost) ...
        # ... (importante: añadir la validación de restricciones aquí) ...
        makespan_val = None
        assignment = {}
        task_completion_days = {}
        objective_val = None # Valor QUBO (energía)
        best_sample = None
        error_message = None
        final_status = status # Empezar con el estado del solver

        if status in ["Feasible", "No Solution Found"] and hasattr(self, 'sampleset') and self.sampleset is not None and len(self.sampleset) > 0:
            try:
                best_sample = self.sampleset.first.sample
                objective_val = self.sampleset.first.energy
                print(f"DEBUG QUBO: Mejor energía encontrada: {objective_val}")

                # 1. Decodificar Makespan
                makespan_val = 0
                if 'makespan_bits' in self.variables:
                    for i, bit_var in self.variables['makespan_bits'].items():
                        var_name = bit_var.variables[0]
                        if best_sample.get(var_name, 0) == 1:
                            makespan_val += 2**i

                # 2. Decodificar EndDay
                if 'end_day_bits' in self.variables:
                    for (p, t), bits in self.variables['end_day_bits'].items():
                        day_val = 0
                        for i, bit_var in bits.items():
                            var_name = bit_var.variables[0]
                            if best_sample.get(var_name, 0) == 1: day_val += 2**i
                        task_completion_days[(p, t)] = day_val

                # 3. Decodificar Asignación (y)
                if 'y_bits' in self.variables:
                    for (p, t, r, d), bits in self.variables['y_bits'].items():
                        hours_val = 0
                        for i, bit_var in bits.items():
                            var_name = bit_var.variables[0]
                            if best_sample.get(var_name, 0) == 1: hours_val += 2**i
                        if hours_val > 1e-4:
                            assignment[(p, t, r, d)] = hours_val

                # 4. Validar Restricciones (Ejemplo simple: horas totales)
                #    ¡SE NECESITA VALIDACIÓN COMPLETA!
                validation_errors = []
                for p in self.project_names:
                     for t in self.tasks_per_project_name.get(p,[]):
                          task_key = (p,t)
                          H_pt = self.hours_required_dict.get(task_key, 0)
                          total_assigned_h = sum(h for (ip, it, r, d), h in assignment.items() if ip==p and it==t)
                          if abs(total_assigned_h - H_pt) > 0.1: # Tolerancia
                               validation_errors.append(f"Violación Horas Totales Tarea {p}-{t}: Req={H_pt}, Asign={total_assigned_h:.1f}")
                # ... añadir validaciones para asignación única, expertise, disponibilidad, secuencia, deadline ...

                if validation_errors:
                     final_status = "Feasible (Violations Found)"
                     error_message = "\n".join(validation_errors)
                     print(f"WARN QUBO: Se encontraron violaciones en la solución: {error_message}")
                elif status == "Feasible": # Si pasó la validación básica y el solver dijo Feasible
                     final_status = "Feasible (Validated Approx)" # Indicar que la validación fue parcial


            except Exception as e:
                import traceback
                print(f"Error extrayendo/decodificando resultados QUBO: {e}\n{traceback.format_exc()}")
                final_status = "Error Extracting"
                error_message = f"Error en extracción/decodificación: {e}"
                makespan_val, assignment, task_completion_days, objective_val = None, {}, {}, None

        # Calcular coste total
        total_cost = None
        if assignment and hasattr(self, 'cost_dict'):
            try:
                total_cost = sum(self.cost_dict.get(r, 0.0) * h for (_, _, r, _), h in assignment.items())
            except Exception as e: print(f"Error calculando coste: {e}")

        self.result = OptimizationResult(
            status=final_status, # Usar el estado final tras validación
            solver_runtime=self.solver_runtime,
            objective_value=objective_val,
            makespan=makespan_val,
            total_cost=total_cost,
            assignment=assignment,
            task_completion_days=task_completion_days,
            error_message=error_message
        )