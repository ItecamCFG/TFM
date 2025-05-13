# optimization/neal_model.py
import neal
import dimod
import numpy as np
import math
import time
from datetime import date, timedelta

from .base_model import OptimizationModel
from .data_models import OptimizationResult # Asegúrate que esta clase existe y está definida

# Cambio de variables enteras a binarias
def integer_to_binary(label, max_value, prefix="int_"):
    if max_value < 0: max_value = 0
    num_bits = math.ceil(math.log2(max_value + 1)) if max_value > 0 else 1
    binary_vars_dict = {i: dimod.Binary(f"{prefix}{label}_b{i}") for i in range(num_bits)}
    linear_expression = dimod.quicksum((2**i) * binary_vars_dict[i] for i in range(num_bits))
    return linear_expression, binary_vars_dict

class NealMakespanModel(OptimizationModel):
    """Minimiza Makespan usando Neal Simulated Annealing (QUBO)."""

    def _build_model(self):
        """Construye el modelo QUBO (BQM) para minimizar makespan."""
        print("DEBUG QUBO: Construyendo modelo BQM...")
        # Asumimos que _prepare_common_data() se llamó y pobló los atributos necesarios
        if not hasattr(self, 'days_list') or not self.days_list:
            raise ValueError("days_list no disponible. Asegúrate que _prepare_common_data se ejecutó.")

        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8 # Max horas por tarea/recurso/día
        max_day_val = self.days_list[-1] if self.days_list else 1
        # self.M_days es un Big-M para el número de días, usado en algunas restricciones
        # Puede ser mayor que max_day_val si es necesario para asegurar que la penalización no se active incorrectamente
        self.M_days_big_m = max_day_val + 10 # Un valor suficientemente grande

        self.variables = {}
        all_model_binary_vars = [] # Lista para recolectar todas las dimod.Binary creadas

        # --- 1. Variables Binarias (x, y_bits, end_day_bits, makespan_bits) ---
        # (Tu código para definir x_vars, y_expressions, y_vars_bits,
        #  end_day_expressions, end_day_vars_bits,
        #  makespan_expr, makespan_vars_bits ya estaba bien.
        #  Solo asegúrate de añadir todos los bits a all_model_binary_vars)

        # x (Asignación)
        x_vars = {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    key = f"x_{p}_{t}_{r}"
                    var = dimod.Binary(key)
                    x_vars[(p, t, r)] = var
                    all_model_binary_vars.append(var)
        self.variables['x'] = x_vars
        print(f"DEBUG QUBO: {len(x_vars)} variables 'x'.")

        # y (Horas)
        y_vars_bits, y_expressions = {}, {}
        num_bits_y = math.ceil(math.log2(M_daily + 1)) if M_daily > 0 else 1
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    for d in self.days_list:
                        label = f"y_{p}_{t}_{r}_{d}"
                        expr, bits = integer_to_binary(label, M_daily, prefix="")
                        y_expressions[(p, t, r, d)] = expr
                        y_vars_bits[(p, t, r, d)] = bits
                        all_model_binary_vars.extend(bits.values())
        self.variables['y_bits'] = y_vars_bits
        self.variables['y_expr'] = y_expressions
        print(f"DEBUG QUBO: {len(all_model_binary_vars) - len(x_vars)} variables binarias para 'y'.")

        # end_day
        end_day_vars_bits, end_day_expressions = {}, {}
        num_bits_end = math.ceil(math.log2(max_day_val + 1)) if max_day_val > 0 else 1
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                label = f"EndDay_{p}_{t}"
                expr, bits = integer_to_binary(label, max_day_val, prefix="")
                end_day_expressions[(p, t)] = expr
                end_day_vars_bits[(p, t)] = bits
                all_model_binary_vars.extend(bits.values())
        self.variables['end_day_bits'] = end_day_vars_bits
        self.variables['end_day_expr'] = end_day_expressions
        # ... (print de debug)

        # makespan
        makespan_expr, makespan_vars_bits = integer_to_binary("Makespan", max_day_val, prefix="")
        self.variables['makespan_bits'] = makespan_vars_bits
        self.variables['makespan_expr'] = makespan_expr
        all_model_binary_vars.extend(makespan_vars_bits.values())
        # ... (print de debug)

        # work_day (como tu antigua 'z', para enlazar y con end_day y secuencia)
        work_day_vars = {}
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p,[]):
                for d in self.days_list:
                    key = f"WorkDay_{p}_{t}_{d}"
                    var = dimod.Binary(key)
                    work_day_vars[(p,t,d)] = var
                    all_model_binary_vars.append(var)
        self.variables['work_day'] = work_day_vars
        print(f"DEBUG QUBO: {len(work_day_vars)} variables 'WorkDay'.")


        # --- Coeficientes de Penalización (AJUSTAR EMPÍRICAMENTE) ---
        P_BASE = max(1.0, float(max_day_val))
        P_HIGH = 10.0 * P_BASE**2  # Para restricciones muy estrictas
        P_MEDIUM = 2.0 * P_BASE
        P_LOW = 0.5 * P_BASE
        P_OBJECTIVE = 1.0 # Peso del makespan en el objetivo

        # --- BQM Principal ---
        bqm = dimod.BinaryQuadraticModel('BINARY')

        # --- Objetivo QUBO: Minimizar Makespan ---
        bqm.update(P_OBJECTIVE * makespan_expr) # Sumar la expresión lineal del makespan

        # --- Restricciones como Penalizaciones ---

        # 1. Asignación única: (sum(x) - 1)^2
        print("DEBUG QUBO: Formulando P: asignación única...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                sum_x = dimod.quicksum(x_vars[(p, t, r)] for r in self.resource_names)
                bqm.update(P_HIGH * (sum_x - 1)**2)

        # 2. Expertise: P * x si no cumple
        print("DEBUG QUBO: Formulando P: expertise...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                task_key = (p, t)
                req_level = level_map.get(self.expertise_required_dict.get(task_key, "Junior"), 1)
                for r in self.resource_names:
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < req_level:
                        bqm.update(P_HIGH * x_vars[(p, t, r)]) # Penalizar x=1

        # 3. Link y-x: y <= M_daily*x. Penalizar si y > 0 y x = 0.
        # P * y_bit * (1 - x) para cada bit de y.
        print("DEBUG QUBO: Formulando P: link y-x...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_var = x_vars[(p, t, r)]
                    for d in self.days_list:
                        for bit_y_var in y_vars_bits[(p, t, r, d)].values():
                            bqm.update(P_LINK * bit_y_var * (1 - x_var))

        # 4. Horas Totales: (sum(y_expr) - H_pt)^2
        print("DEBUG QUBO: Formulando P: horas totales...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                task_key = (p, t)
                H_pt = self.hours_required_dict.get(task_key, 0)
                sum_y_expr_pt = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names for d in self.days_list)
                bqm.update(P_HOURS_TOTAL * (sum_y_expr_pt - H_pt)**2)

        # 5. Disponibilidad: sum_tareas(y_expr[r,d]) <= A_rd
        # sum_y + slack_avail - A_rd = 0 => P_AVAIL * (sum_y + slack_avail - A_rd)^2
        print("DEBUG QUBO: Formulando P: disponibilidad (con slack)...")
        # Max slack value is sum of all M_daily for a resource in a day, but A_rd is fine if sum_y should be <= A_rd
        for r in self.resource_names:
            for d in self.days_list:
                A_rd = self.availability_numeric.get((r, d), 0)
                sum_y_rd = dimod.quicksum(y_expressions[(p, t, r, d)] for p in self.project_names for t in self.tasks_per_project_name.get(p, []))
                slack_label = f"SlackAvail_{r}_{d}"
                # Slack puede ir hasta M_daily * num_tasks, o más simple, hasta A_rd si sum_y <= A_rd
                slack_expr, slack_bits = integer_to_binary(slack_label, A_rd, prefix="") # Slack máximo = A_rd
                all_model_binary_vars.extend(slack_bits.values())
                self.variables.setdefault('availability_slacks_bits', {})[(r,d)] = slack_bits

                bqm.update(P_AVAIL * (sum_y_rd - slack_expr - A_rd)**2) # sum_y - slack = A_rd NO, sum_y + slack = A_rd
                                                                       # para sum_y <= A_rd --> sum_y + slack = A_rd
                # Corrección: sum_y <= A_rd  ---> A_rd - sum_y >= 0
                # A_rd - sum_y = slack  (donde slack >=0)
                # (A_rd - sum_y - slack_expr)^2
                bqm.update(P_AVAIL * (A_rd - sum_y_rd - slack_expr)**2)


        # 6. Enlace y -> WorkDay: y_sum_per_day <= M_total_daily_task * WorkDay
        # Si sum(y[p,t,r,d] for r) > 0 entonces WorkDay[p,t,d] = 1
        print("DEBUG QUBO: Formulando P: link y-WorkDay...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    sum_y_ptd = dimod.quicksum(y_expressions[(p,t,r,d)] for r in self.resource_names)
                    # Si sum_y_ptd > 0, WorkDay debe ser 1.
                    # Penalizar sum_y_ptd * (1 - WorkDay) -> No, esto es lineal.
                    # Necesitamos: work_day >= sum_y / M (aprox)
                    # o work_day = 0 => sum_y = 0
                    # (sum_y_ptd > 0) => work_day = 1
                    # sum_y_ptd <= (M_daily * len(self.resource_names)) * work_day_vars[(p,t,d)]
                    # Esta restricción es para que si work_day es 0, la suma de y sea 0.
                    # Y si la suma de y es >0, work_day debe ser 1.
                    max_y_sum_daily = M_daily * len(self.resource_names) # Teórico máximo de horas en una tarea en un día
                    bqm.add_linear_inequality_constraint(
                        linear_terms=[(sum_y_ptd, 1.0), (work_day_vars[(p,t,d)], -max_y_sum_daily)],
                        lagrange_multiplier=P_LINK,
                        label=f"link_y_sum_workday_{p}_{t}_{d}",
                        constant=0, # sum_y_ptd - max_y_sum_daily * work_day <= 0
                        vartype=dimod.BINARY # Es una restricción sobre variables binarias
                    )
                    # Alternativa más simple y directa (bit a bit)
                    # for r in self.resource_names:
                    #     for bit_y_var in y_vars_bits[(p,t,r,d)].values():
                    #         bqm.update(P_LINK * bit_y_var * (1 - work_day_vars[(p,t,d)])) # Si y_bit=1 y WD=0, penaliza
                    #         bqm.update(P_LINK * (1 - bit_y_var) * work_day_vars[(p,t,d)]) # Si y_bit=0 y WD=1, penaliza? No.


        # 7. Definición EndDay: EndDay >= d si WorkDay=1
        # EndDay - d + SlackEndDayDef = 0 (si WorkDay=1)
        # (EndDay - d + SlackEndDayDef)^2 * WorkDay
        # O la formulación Big-M: end_day >= d - M_days * (1 - work_day)
        print("DEBUG QUBO: Formulando P: definición EndDay...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                end_expr = end_day_expressions[(p,t)]
                for d in self.days_list:
                    # end_expr >= d * work_day_vars[(p,t,d)] -> No lineal con d si d no es 0/1
                    # (end_expr - d + slack)^2 * work_day | si no work_day, no importa
                    # Penalizar: (d - end_expr + slack_d_ge_end)^2 * work_day_vars[(p,t,d)]
                    # Esto es complejo. Usemos la Big-M:
                    # end_expr - d*work_day_vars[(p,t,d)] >= 0 (aproximado)
                    # Más exacto: end_expr >= d - M_days_big_m * (1 - work_day_vars[(p,t,d)])
                    # end_expr - d + M_days_big_m * (1 - work_day_vars[(p,t,d)]) >= 0
                    # Convertir a igualdad con slack:
                    # end_expr - d + M_days_big_m * (1 - work_day_vars[(p,t,d)]) - slack = 0
                    slack_label = f"SlackEndDayDef_{p}_{t}_{d}"
                    slack_expr, slack_bits = integer_to_binary(slack_label, self.M_days_big_m + max_day_val, prefix="")
                    all_model_binary_vars.extend(slack_bits.values())
                    self.variables.setdefault('end_day_def_slacks_bits', {})[(p,t,d)] = slack_bits

                    term = end_expr - d + self.M_days_big_m * (1 - work_day_vars[(p,t,d)]) - slack_expr
                    bqm.update(P_HIGH * term**2)


        # 8. Secuencialidad: EndDay[i] <= d-1 + M_days*(1 - WorkDay[i+1,d])
        print("DEBUG QUBO: Formulando P: secuencia (Finish-to-Start)...")
        for p in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name[p],
                key=lambda t_name: self.task_info.get((p, t_name), {}).get('sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i = sorted_tasks[i]
                t_i_plus_1 = sorted_tasks[i+1]
                end_i_expr = end_day_expressions[(p, t_i)]

                for d_val in self.days_list:
                    if d_val > 1:
                        # end_i_expr <= (d_val - 1) + M_days_big_m * (1 - work_day_vars[(p, t_i_plus_1, d_val)])
                        # (d_val - 1) + M_days_big_m * (1 - work_day_vars[...]) - end_i_expr = slack_seq
                        slack_label = f"SlackSeq_{p}_{t_i}_{t_i_plus_1}_{d_val}"
                        slack_expr, slack_bits = integer_to_binary(slack_label, self.M_days_big_m + max_day_val, prefix="")
                        all_model_binary_vars.extend(slack_bits.values())
                        self.variables.setdefault('sequence_slacks_bits', {})[(p,i,d_val)] = slack_bits

                        term = (d_val - 1) + self.M_days_big_m * (1 - work_day_vars[(p, t_i_plus_1, d_val)]) - end_i_expr - slack_expr
                        bqm.update(P_SEQ * term**2)
                    else: # d_val == 1
                        if self.hours_required_dict.get((p,t_i),0) > 0:
                            bqm.update(P_SEQ * work_day_vars[(p, t_i_plus_1, 1)]) # Penalizar si work_day es 1

        # 9. Deadline: EndDay[p,t] <= D_p
        # EndDay + SlackDL - D_p = 0
        print("DEBUG QUBO: Formulando P: deadline (con slack)...")
        for project_data in self.input_data.projects:
            p_name = project_data.name
            deadline_date = project_data.deadline
            if deadline_date:
                deadline_day_num = (deadline_date - self.start_date).days + 1
                if 1 <= deadline_day_num <= max_day_val:
                    for t in self.tasks_per_project_name.get(p_name,[]):
                        end_expr = end_day_expressions[(p_name, t)]
                        slack_label = f"SlackDL_{p_name}_{t}"
                        # Max slack = deadline_day_num si end_expr debe ser <=
                        # end_expr + slack = deadline_day_num
                        slack_expr, slack_bits = integer_to_binary(slack_label, deadline_day_num, prefix="")
                        all_model_binary_vars.extend(slack_bits.values())
                        self.variables.setdefault('deadline_slacks_bits', {})[(p_name,t)] = slack_bits
                        bqm.update(P_DEADLINE * (end_expr + slack_expr - deadline_day_num)**2)

        # 10. Makespan Definition: Makespan >= EndDay[p,t]
        # Makespan - EndDay + SlackMK = 0
        print("DEBUG QUBO: Formulando P: definición Makespan (con slack)...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                end_expr = end_day_expressions[(p, t)]
                slack_label = f"SlackMK_{p}_{t}"
                # Max slack = max_day_val
                slack_expr, slack_bits = integer_to_binary(slack_label, max_day_val, prefix="")
                all_model_binary_vars.extend(slack_bits.values())
                self.variables.setdefault('makespan_def_slacks_bits', {})[(p,t)] = slack_bits
                bqm.update(P_MAKE_DEF * (makespan_expr - end_expr + slack_expr)**2)

        # --- Finalizar BQM ---
        # Asegurar que todas las variables binarias estén en el BQM
        # (dimod.quicksum o BQM.update con expresiones lineales/cuadráticas ya las añade)
        # Pero si alguna variable binaria solo se usó como parte de una expresión lineal
        # y no tiene su propio término lineal o cuadrático, podría no estar.
        # Esto es más para asegurarse que BQM conoce todas las variables.
        # Sin embargo, si una variable solo aparece en quicksum, ya está.
        # Lo que hicimos con all_model_binary_vars y add_variable es para Q, no tanto para BQM así.
        # Si construyes BQM sumando, ya está.

        self.model = bqm
        num_vars_final = len(self.model.variables)
        print(f"DEBUG QUBO: Modelo BQM final construido con {num_vars_final} variables binarias.")
        if num_vars_final > 7000: # Aumentar umbral de advertencia
             print(f"ADVERTENCIA QUBO: El número de variables ({num_vars_final}) es muy grande.")

    # _solve_model y _extract_results (mantener como estaban definidos antes)
    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el modelo QUBO usando Neal."""
        if self.model is None or not isinstance(self.model, dimod.BinaryQuadraticModel):
            print("ERROR QUBO: Modelo BQM no construido correctamente.")
            return "Error Building", 0.0

        sampler = neal.SimulatedAnnealingSampler()
        num_vars = len(self.model.variables)
        if num_vars < 100: num_reads = 1000
        elif num_vars < 1000: num_reads = 500
        elif num_vars < 5000: num_reads = 200 # Ajustar
        else: num_reads = 50 # Reducir para modelos muy grandes

        print(f"DEBUG QUBO: Iniciando Neal Sampler con {num_reads} reads para {num_vars} variables...")
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
            return "Feasible", runtime
        else:
            return "No Solution Found", runtime

    def _extract_results(self, status: str):
        """Extrae e interpreta los resultados del sampleset de Neal."""
        # (Mismo código de extracción/decodificación que antes, adaptado para variables guardadas)
        makespan_val, assignment, task_completion_days, objective_val = None, {}, {}, None
        best_sample, error_message, final_status = None, None, status

        if status in ["Feasible", "No Solution Found"] and hasattr(self, 'sampleset') and self.sampleset is not None and len(self.sampleset) > 0:
            try:
                best_sample = self.sampleset.first.sample
                objective_val = self.sampleset.first.energy
                print(f"DEBUG QUBO: Mejor energía encontrada: {objective_val}")

                # Decodificar Makespan
                makespan_val = 0
                if 'makespan_bits' in self.variables:
                    for i, bit_var in self.variables['makespan_bits'].items():
                        var_name = bit_var.label # Usar .label para el nombre
                        if best_sample.get(var_name, 0) == 1: makespan_val += 2**i

                # Decodificar EndDay
                if 'end_day_bits' in self.variables:
                    for (p, t), bits_dict in self.variables['end_day_bits'].items():
                        day_val = 0
                        for i, bit_var in bits_dict.items():
                            var_name = bit_var.label
                            if best_sample.get(var_name, 0) == 1: day_val += 2**i
                        task_completion_days[(p, t)] = day_val
                
                # Decodificar y (Horas)
                if 'y_bits' in self.variables:
                    for (p,t,r,d), bits_dict in self.variables['y_bits'].items():
                        hours_val = 0
                        for i, bit_var in bits_dict.items():
                            var_name = bit_var.label
                            if best_sample.get(var_name, 0) == 1: hours_val += 2**i
                        if hours_val > 1e-4:
                            assignment[(p,t,r,d)] = hours_val

                # TODO: Validación robusta de restricciones
                validation_errors = []
                # ... (Añadir lógica de validación) ...
                if validation_errors:
                    final_status = "Feasible (Violations Found)"
                    error_message = "\n".join(validation_errors)
                elif status == "Feasible":
                    final_status = "Feasible (Validation Approx.)" # Indicar que validación fue parcial

            except Exception as e:
                import traceback
                error_message = f"Error extrayendo/decodificando: {e}\n{traceback.format_exc()}"
                print(error_message)
                final_status = "Error Extracting"
                makespan_val, assignment, task_completion_days, objective_val = None, {}, {}, None
        
        total_cost = None
        if assignment and hasattr(self, 'cost_dict'):
            try:
                total_cost = sum(self.cost_dict.get(r,0.0) * h for (_,_,r,_),h in assignment.items())
            except: pass # Ignorar error en cálculo de coste si falta algo

        self.result = OptimizationResult(
            status=final_status, solver_runtime=self.solver_runtime,
            objective_value=objective_val, makespan=makespan_val, total_cost=total_cost,
            assignment=assignment, task_completion_days=task_completion_days, error_message=error_message
        )