# optimization/neal_model.py
import neal
import dimod
import numpy as np
import math
import time
# from itertools import product # Ya no se usa product directo con get(p, [])

from optimization.base_model import OptimizationModel
from optimization.data_models import OptimizationResult # Asumiendo que tienes esta clase

# Helper para codificación binaria de enteros
def integer_to_binary(label, max_value, BQM_or_Q_dict=None, prefix="int_"):
    """Crea variables binarias para representar un entero y devuelve la expresión lineal."""
    # max_value debe ser >= 0. Si es 0, representa 0 (1 bit, 0).
    num_bits = math.ceil(math.log2(max_value + 1)) if max_value > 0 else 1
    binary_vars = {i: dimod.Binary(f"{prefix}{label}_b{i}") for i in range(num_bits)}
    # Expresión lineal: sum(2^i * var_i)
    linear_expression = dimod.quicksum( (2**i) * binary_vars[i] for i in range(num_bits) )
    # Nota: No añadir variables al BQM/Q_dict aquí directamente. Se hace después de crear todas las variables.

    return linear_expression, binary_vars # Devolver la expresión y las variables individuales

class NealMakespanModel(OptimizationModel):
    """Minimiza Makespan usando Neal Simulated Annealing (QUBO)."""

    def _build_model(self):
        """Construye el modelo QUBO (BQM) para minimizar makespan."""
        print("DEBUG QUBO: Construyendo modelo BQM...")
        # Datos preparados en self por _prepare_common_data
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8 # Límite para y (si lo codificamos) - Horas máximas por día por
        # Calcular max_day del planning horizon, asegurando que sea al menos 1 si days_list está vacío
        max_day = max(1, self.days_list[-1] if self.days_list else 1)


        # --- Variables Binarias ---
        self.variables = {} # Reiniciar diccionario de variables por tipo

        # 1. x (Asignación): x[p, t, r] = 1 si el recurso r está asignado a la tarea t del proyecto p
        x_vars = {}
        # Correcto bucle anidado
        for p in self.project_names:
            tasks_for_p = self.tasks_per_project_name.get(p, [])
            for t in tasks_for_p:
                for r in self.resource_names:
                    # Clave compuesta única para la variable binaria
                    key = f"x_{p}_{t}_{r}"
                    x_vars[(p, t, r)] = dimod.Binary(key)
        self.variables['x'] = x_vars
        print(f"DEBUG QUBO: {len(x_vars)} variables 'x' creadas.")

        # 2. y (Horas): y[p, t, r, d] = Horas trabajadas por r en t de p en el día d (codificado binariamente)
        y_vars_bits = {} # Guarda los diccionarios de bits individuales por (p,t,r,d)
        y_expressions = {} # Guarda las expresiones lineales (el valor entero decodificado)
        num_bits_y = math.ceil(math.log2(M_daily + 1)) if M_daily > 0 else 1
        print(f"DEBUG QUBO: Codificando 'y' (horas 0-{M_daily}) usando {num_bits_y} bits...")
        count_y_vars = 0
        # Correcto bucle anidado
        for p in self.project_names:
            tasks_for_p = self.tasks_per_project_name.get(p, [])
            for t in tasks_for_p:
                for r in self.resource_names:
                     for d in self.days_list:
                         label = f"y_{p}_{t}_{r}_{d}"
                         expr, bits = integer_to_binary(label, M_daily) # No pasar BQM/Q_dict aquí
                         y_expressions[(p, t, r, d)] = expr
                         y_vars_bits[(p, t, r, d)] = bits # Guardar el diccionario {bit_idx: Binary(...)}
                         count_y_vars += num_bits_y # Contar el número total de variables binarias para y

        self.variables['y_bits'] = y_vars_bits # Almacenar el diccionario de diccionarios de bits
        self.variables['y_expr'] = y_expressions # Guardar expresiones lineales
        print(f"DEBUG QUBO: {count_y_vars} variables binarias para 'y' creadas.")

        # 3. end_day: end_day[p, t] = Día de finalización de la tarea t del proyecto p (codificado binariamente)
        end_day_vars_bits = {} # Guarda los diccionarios de bits individuales por (p,t)
        end_day_expressions = {} # Guarda las expresiones lineales
        num_bits_end = math.ceil(math.log2(max_day + 1)) if max_day > 0 else 1
        print(f"DEBUG QUBO: Codificando 'end_day' (días 1-{max_day}) usando {num_bits_end} bits...")
        count_end_vars = 0
        # Correcto bucle anidado
        for p in self.project_names:
             tasks_for_p = self.tasks_per_project_name.get(p, [])
             for t in tasks_for_p:
                 label = f"EndDay_{p}_{t}"
                 expr, bits = integer_to_binary(label, max_day) # No pasar BQM/Q_dict aquí
                 end_day_expressions[(p, t)] = expr
                 end_day_vars_bits[(p, t)] = bits # Guardar el diccionario {bit_idx: Binary(...)}
                 count_end_vars += num_bits_end # Contar el número total de variables binarias para end_day

        self.variables['end_day_bits'] = end_day_vars_bits # Almacenar el diccionario de diccionarios de bits
        self.variables['end_day_expr'] = end_day_expressions # Guardar expresiones lineales
        print(f"DEBUG QUBO: {count_end_vars} variables binarias para 'end_day' creadas.")

        # 4. makespan: makespan = Día de finalización del último proyecto (codificado binariamente)
        print(f"DEBUG QUBO: Codificando 'makespan' (días 1-{max_day}) usando {num_bits_end} bits...")
        makespan_expr, makespan_vars_bits = integer_to_binary("Makespan", max_day) # No pasar BQM/Q_dict aquí
        self.variables['makespan_bits'] = makespan_vars_bits # Almacenar el diccionario {bit_idx: Binary(...)}
        self.variables['makespan_expr'] = makespan_expr # Guardar expresión lineal
        print(f"DEBUG QUBO: {num_bits_end} variables binarias para 'makespan' creadas.")


        # --- Coeficientes de Penalización (¡REQUIEREN AJUSTE!) ---
        # Estos valores son cruciales y difíciles de determinar a priori.
        # Deben ser mayores que cualquier posible cambio en el objetivo real.
        # Ajusta estos valores empíricamente. Pueden necesitar ser escalados.
        P_ASSIGN = 1000.0 * (max_day + 1) # Penalización alta para asignación única/expertise
        P_HOURS = 10.0 # Penalización por horas (relacionada con la necesidad vs lo asignado)
        P_AVAIL = 100.0 # Penalización disponibilidad diaria por recurso
        P_SEQ = 500.0 * (max_day + 1) # Penalización secuencia de tareas
        P_DEADLINE = 1000.0 * (max_day + 1) # Penalización incumplimiento deadline de proyecto
        P_MAKE = 1.0 # Coeficiente para el objetivo makespan (generalmente 1.0 o mayor si makespan es muy crítico)


        # --- Objetivo QUBO = H_objetivo + sum(P_i * H_restriccion_i) ---
        # H_objetivo: Minimizar makespan (representado por makespan_expr)
        objective_bqm = P_MAKE * makespan_expr

        # H_restricciones: Convertir cada restricción MILP en una penalización cuadrática
        constraints_bqm = dimod.BinaryQuadraticModel('BINARY')

        # Restricción: Suma de horas y relación con x (sum(y_expr for d) == x * H_pt)
        # Penalización: (sum(y_expr for d) - x * H_pt)^2 (Requiere variables auxiliares para el término x * y)
        print("WARN QUBO: Penalización de relación entre x, y y horas totales NO implementada explícitamente (complejo).")
        # for p in self.project_names: ... (Resto del bucle original comentado)


        # Restricción 4.1 Asignación única por tarea: Cada tarea se asigna a EXACTAMENTE un recurso.
        # sum(x[p,t,r] for r) == 1
        # Penalización: (sum(x[p,t,r] for r) - 1)^2
        print("DEBUG QUBO: Formulando penalización asignación única por tarea...")
        # Correcto bucle anidado
        for p in self.project_names:
             tasks_for_p = self.tasks_per_project_name.get(p, [])
             for t in tasks_for_p:
                 constraint_expr = dimod.quicksum(x_vars[(p, t, r)] for r in self.resource_names) - 1
                 constraints_bqm.update((P_ASSIGN * constraint_expr**2))


        # Restricción 4.1 Expertise: Si x[p,t,r]=1, res_level(r) >= req_level(p,t).
        # Penalización: P * x[p,t,r] si res_level(r) < req_level(p,t)
        print("DEBUG QUBO: Formulando penalización expertise...")
        # Correcto bucle anidado
        for p in self.project_names:
             tasks_for_p = self.tasks_per_project_name.get(p, [])
             for t in tasks_for_p:
                 task_key = (p,t)
                 # Manejar caso donde task_key no está en expertise_required_dict, asumir Junior (nivel 1)
                 required_level = level_map.get(self.expertise_required_dict.get(task_key, "Junior"), 1)
                 for r in self.resource_names:
                      # Manejar caso donde resource r no está en expertise_dict, asumir Junior (nivel 1)
                      res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                      if res_level < required_level:
                           # Si el recurso NO cumple el requisito, penalizamos si x[p,t,r] es 1
                           constraints_bqm.update(P_ASSIGN * x_vars[(p, t, r)]) # Penalizar si x=1 cuando no debe


        # Restricción 4.1 Horas Totales por Tarea (usando y): sum(y[p,t,r,d] for r, d) == HorasRequeridas(p,t)
        # Penalización: (sum(y_expr[p,t,r,d] for r, d) - H_pt)^2
        print("DEBUG QUBO: Formulando penalización horas totales por tarea (Suma de y)...")
        # Correcto bucle anidado
        for p in self.project_names:
             tasks_for_p = self.tasks_per_project_name.get(p, [])
             for t in tasks_for_p:
                  task_key = (p,t)
                  H_pt = self.hours_required_dict.get(task_key, 0)
                  # Suma de las expresiones lineales de horas para esta tarea a lo largo de todos los recursos y días
                  sum_y_expr_pt = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names for d in self.days_list)
                  constraint_expr = sum_y_expr_pt - H_pt
                  constraints_bqm.update((P_HOURS * constraint_expr**2))


        # Restricción 4.2 Disponibilidad: sum(y[p,t,r,d] for p,t) <= DisponibilidadDiaria(r,d)
        print("WARN QUBO: Penalización de disponibilidad diaria NO implementada explícitamente (complejo).")
        # Implementación pendiente...


        # Restricción 4.4 Secuencia: Si la tarea t_i debe ir antes que t_{i+1} en el proyecto p, entonces EndDay(p, t_i) < StartDay(p, t_{i+1}).
        print("WARN QUBO: Penalización de secuencia finish-to-start NO implementada explícitamente (complejo).")
        # Implementación pendiente...


        # Restricción 4.5 Deadline: EndDay(p, t) <= DeadlineDay(p) para todas las tareas t del proyecto p.
        print("WARN QUBO: Penalización de deadline NO implementada explícitamente (complejo).")
        # Implementación pendiente...


        # Restricción 4.6 Makespan definition: makespan >= EndDay(p, t) para todas las tareas t de todos los proyectos p.
        print("WARN QUBO: Penalización definición makespan NO implementada explícitamente (complejo).")
        # Correcto bucle anidado (aunque la penalización está vacía)
        # for p in self.project_names: ... (Resto del bucle original comentado)
        pass # Implementación pendiente...


        # --- Combinar Objetivo y Restricciones ---
        # Combinar el objetivo con todas las penalizaciones de restricciones
        # Asegúrate de que todas las variables utilizadas en las restricciones se hayan creado y añadido al BQM
        # dimod.BinaryQuadraticModel.from_qubo(Q) es una forma si construiste Q.
        # Si construyes sub-BQMs (como objective_bqm y constraints_bqm), puedes sumarlos.
        # La suma de BQMs de dimod combina las variables y coeficientes automáticamente.

        final_bqm = objective_bqm + constraints_bqm

        # --- CORRECCIÓN DEL ERROR AQUÍ ---
        # Recolectar todas las variables binarias creadas para añadirlas al BQM final
        all_binary_variables = []
        # Añadir variables x
        all_binary_variables.extend(self.variables['x'].values())

        # Añadir variables y_bits
        for bits_dict_y in self.variables['y_bits'].values():
            # bits_dict_y es un diccionario {bit_idx: Binary(...)}
            all_binary_variables.extend(bits_dict_y.values())

        # Añadir variables end_day_bits
        for bits_dict_end in self.variables['end_day_bits'].values():
             # bits_dict_end es un diccionario {bit_idx: Binary(...)}
             all_binary_variables.extend(bits_dict_end.values())

        # Añadir variables makespan_bits (ya es un diccionario {bit_idx: Binary(...)})
        all_binary_variables.extend(self.variables['makespan_bits'].values())

        # Añadir todas las variables binarias recolectadas al BQM
        # dimod.add_variables_from_sampleset() o BQM.add_variable()
        # Añadir con coeficiente lineal 0.0 si no existen ya en el BQM sumado
        for bit_var in all_binary_variables:
             # bit_var es un objeto dimod.Binary como Binary('x_p_t_r') o Binary('y_p_t_r_d_b0')
             var_name = bit_var.variables[0] # Obtener el nombre de la variable
             # add_variable(v, bias) añade si no existe o actualiza el bias. Si ya está, el bias sumado se mantiene.
             final_bqm.add_variable(var_name, 0.0)


        self.model = final_bqm # Guardar el BQM construido

        print(f"DEBUG QUBO: Modelo BQM construido con {len(self.model.variables)} variables binarias.")
        print(f"DEBUG QUBO: Tiene {len(self.model.linear)} términos lineales y {len(self.model.quadratic)} términos cuadráticos.")
        # print("DEBUG QUBO: Variables:", self.model.variables) # Descomentar para ver todas las variables


    def _solve_model(self) -> tuple[str, float]:
        """Resuelve el modelo QUBO usando Neal."""
        if self.model is None or not isinstance(self.model, dimod.BinaryQuadraticModel):
             print("ERROR QUBO: Modelo BQM no construido correctamente.")
             return "Error", 0.0

        sampler = neal.SimulatedAnnealingSampler()
        # Configurar num_reads y potencialmente otros parámetros de Neal
        num_reads = 100 # Número de intentos de annealing (ajustable)
        # Puedes usar self.input_data.config.solver_time_limit para derivar num_reads o usar sample_qc si usas D-Wave Leap
        # Por ahora, num_reads es fijo.
        print(f"DEBUG QUBO: Iniciando Neal Sampler con {num_reads} reads...")
        start_t = time.time()
        try:
            # Usar sample(bqm) con el BQM construido
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
             # Neal no da un status como "Optimal", simplemente devuelve muestras.
             # Consideraremos "Feasible" si hay muestras.
             # Una solución "válida" en QUBO es una muestra que satisface (tiene energía baja) todas las restricciones.
             # La verdadera factibilidad solo se puede verificar *después* de la decodificación.
             return "Feasible", runtime
        else:
             # Esto puede significar que no se encontraron muestras o que todas las energías son muy altas
             return "No Solution Found", runtime


    def _extract_results(self, status: str):
        """Extrae e interpreta los resultados del sampleset de Neal."""
        makespan_val = None
        assignment = {} # { (p, t, r, d): hours }
        task_completion_days = {} # { (p, t): day_number }
        objective_val = None # Valor QUBO (energía de la mejor muestra)
        best_sample = None
        error_message = None

        # Si hubo error en el solver, el estado ya reflejará eso.
        # Solo intentamos extraer si el estado del solver fue "Feasible" o similar.
        # También incluimos "No Solution Found" para intentar decodificar si hay sampleset pero vacío o con alta energía
        if status in ["Feasible", "Feasible (Validation Pending)", "No Solution Found"] and self.sampleset is not None and len(self.sampleset) > 0:
            try:
                # Neal devuelve un SampleSet. La primera muestra suele ser la de menor energía.
                best_sample = self.sampleset.first.sample
                objective_val = self.sampleset.first.energy # Energía de la mejor muestra

                print(f"DEBUG QUBO: Mejor energía encontrada: {objective_val}")
                # print(f"DEBUG QUBO: Mejor muestra: {best_sample}") # Puede ser muy largo

                # --- Decodificar Resultados ---
                # Es complejo y propenso a errores si las penalizaciones no funcionaron perfectamente.

                # 1. Decodificar Makespan
                makespan_val = 0
                if 'makespan_bits' in self.variables:
                     # self.variables['makespan_bits'] es el diccionario {bit_idx: Binary(...)}
                     for i, bit_var in self.variables['makespan_bits'].items():
                          # Acceder al valor de la variable binaria en la muestra
                          var_name = bit_var.variables[0] # Obtener nombre de la variable
                          if best_sample.get(var_name, 0) == 1: # Usar .get con default 0 por seguridad
                              makespan_val += 2**i
                # Ajustar Makespan si es 0 y hay tareas planificables (puede ser 0 si el makespan óptimo es 0 días, ej. no hay tareas)
                # if makespan_val == 0 and (self.project_names or self.task_names): makespan_val = 1 # Esto podría ser confuso. Mejor confiar en la decodificación.


                # 2. Decodificar EndDay por Tarea
                if 'end_day_bits' in self.variables:
                     # self.variables['end_day_bits'] es el diccionario {(p,t): {bit_idx: Binary(...)}}
                     for (p, t), bits in self.variables['end_day_bits'].items():
                          day_val = 0
                          # 'bits' es el diccionario {bit_idx: Binary(...)} para la tarea (p,t)
                          for i, bit_var in bits.items():
                               var_name = bit_var.variables[0]
                               if best_sample.get(var_name, 0) == 1:
                                    day_val += 2**i
                          task_completion_days[(p, t)] = day_val
                # print(f"DEBUG QUBO: Task completion days: {task_completion_days}")

                # 3. Decodificar Asignación y Horas (y)
                if 'y_bits' in self.variables:
                     # self.variables['y_bits'] es el diccionario {(p,t,r,d): {bit_idx: Binary(...)}}
                     for (p, t, r, d), bits in self.variables['y_bits'].items():
                          hours_val = 0
                          # 'bits' es el diccionario {bit_idx: Binary(...)} para (p,t,r,d)
                          for i, bit_var in bits.items():
                               var_name = bit_var.variables[0]
                               if best_sample.get(var_name, 0) == 1:
                                    hours_val += 2**i
                          # Guardar solo si las horas decodificadas > 0
                          if hours_val > 1e-4: # Usar una tolerancia para evitar valores muy pequeños
                               assignment[(p, t, r, d)] = hours_val
                # print(f"DEBUG QUBO: Assignment (hours): {assignment}")


                # 4. Validar Restricciones (¡MUY IMPORTANTE!)
                # Aquí deberías añadir código para verificar si la solución decodificada
                # cumple las restricciones originales.
                # Si una restricción tiene alta energía de penalización en la mejor muestra, es probable que no se cumpla.
                # Puedes iterar sobre constraints_bqm.energy(best_sample) para ver la energía de cada componente,
                # pero eso requiere acceso al BQM de restricciones, que no se guarda en self.
                # Una validación basada en los resultados decodificados (assignment, task_completion_days) es mejor.

                # TODO: Implementar validación real de las restricciones decodificadas aquí.
                # Si se encuentran violaciones:
                # status = "Feasible (Violations Found)"
                # error_message = "Mensajes detallados de las violaciones."
                # else:
                # status = "Feasible" # O mantener "Feasible" del solver

                # Por ahora, si llegamos aquí sin error de extracción, asumimos que se encontró una solución,
                # pero marcamos que la validación está pendiente.
                if status == "Feasible": # Si el solver reportó "Feasible"
                     status = "Feasible (Validation Pending)" # Sobrescribir para indicar que falta validar


            except Exception as e:
                # Si hay error durante la decodificación o validación
                import traceback
                print(f"Error extrayendo/decodificando resultados QUBO: {e}\n{traceback.format_exc()}")
                status = "Error Extracting"
                error_message = f"Error en extracción/decodificación: {e}\n{traceback.format_exc()}"
                # Limpiar resultados parciales si hubo error
                makespan_val = None
                assignment = {}
                task_completion_days = {}
                objective_val = None # La energía sigue siendo válida, pero no los resultados decodificados


        # Calcular el coste total si hay asignaciones y costos por recurso
        total_cost = None
        if assignment and hasattr(self, 'cost_dict') and self.cost_dict:
             try:
                 # Sumar el coste por cada hora asignada (asumiendo cost_dict es por hora)
                 total_cost = sum(self.cost_dict.get(r, 0.0) * h for (p, t, r, d), h in assignment.items())
             except Exception as e:
                 print(f"Error calculando coste total: {e}")
                 total_cost = None
                 if error_message:
                     error_message += f"\nError calculando coste: {e}"
                 else:
                      error_message = f"Error calculando coste: {e}"


        # Crear el objeto OptimizationResult
        self.result = OptimizationResult(
            status=status,
            solver_runtime=self.solver_runtime,
            objective_value=objective_val, # Valor de la energía QUBO
            makespan=makespan_val,
            total_cost=total_cost, # Añadir el coste total calculado
            assignment=assignment,
            task_completion_days=task_completion_days, # Puede que no se use directamente en UI Gantt, pero útil
            error_message=error_message # Añadir mensajes de error/validación
            # input_data=self.input_data # No guardar input_data completo en el resultado
        )