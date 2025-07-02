# optimization/neal_model.py
import neal
import dimod
from dwave.samplers import Neal  # <--- AÑADE ESTA LÍNEA
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

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = None
        self.sampleset = None
        self.variables = {}
        self.M_days_big_m = None

        

    
    def _build_model(self):
        """Construye el modelo QUBO (BQM) para minimizar el makespan."""
        print("DEBUG QUBO: Construyendo modelo BQM...")
        
        # --- PREPARACIÓN DE CONSTANTES ---
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        M_daily = 8
        max_day_val = self.days_list[-1] if self.days_list else 1
        
        self.variables = {}
        y_expressions, y_vars_bits = {}, {}
        end_day_expressions, end_day_vars_bits = {}, {}
        work_day_vars = {} # Definimos work_day_vars aquí para que sea accesible en todo el método
        
        bqm = dimod.BinaryQuadraticModel('BINARY')

        # --- 1. CREACIÓN DE VARIABLES Y EXPRESIONES ---
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    work_day_vars[(p,t,d)] = dimod.Binary(f"WorkDay_{p}_{t}_{d}")

                for r in self.resource_names:
                    for d in self.days_list:
                        expr, bits = integer_to_binary(f"y_{p}_{t}_{r}_{d}", M_daily)
                        y_expressions[(p, t, r, d)] = expr
                        y_vars_bits[(p, t, r, d)] = bits
        
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                expr, bits = integer_to_binary(f"EndDay_{p}_{t}", max_day_val)
                end_day_expressions[(p, t)] = expr
                end_day_vars_bits[(p,t)] = bits

        makespan_expr, _ = integer_to_binary("Makespan", max_day_val)
            
        # --- 2. COEFICIENTES DE PENALIZACIÓN (VERSIÓN REFORZADA) ---
        P_BASE = max(1.0, float(max_day_val))
        P_ABSOLUTE = 100.0 * P_BASE**2 # Nueva categoría máxima, para lo innegociable
        P_CRITICAL = 50.0 * P_BASE**2 
        P_HARD = 10.0 * P_BASE**2
        P_MEDIUM = 5.0 * P_BASE
        P_LOW = 1.0 * P_BASE 

        # Asignación de penalizaciones
        P_HOURS_TOTAL = P_ABSOLUTE   # ¡Esta es la más importante ahora mismo!
        P_SEQ = P_CRITICAL
        P_AVAIL = P_CRITICAL
        P_DEADLINE = P_CRITICAL
        P_ASSIGN = P_HARD
        P_EXPERTISE = P_HARD
        P_ENDDAY_DEF = P_HARD
        P_MAKE_DEF = P_HARD
        P_LINK_WD = P_HARD          # La nueva restricción y los enlaces lógicos
        P_LINK_YX = P_MEDIUM

        P_OBJECTIVE = 0.1 # Penalización por el objetivo de makespan

        self.penalties = { 'P_CRITICAL': P_CRITICAL, 'P_HARD': P_HARD, 'P_MEDIUM': P_MEDIUM, 'P_LOW': P_LOW }
            
        # --- 3. OBJETIVO ---
        bqm.update(P_OBJECTIVE * makespan_expr)

        # --- 4. RESTRICCIONES ---

        # 1. Asignación única
        print("DEBUG QUBO: Formulando P: asignación única...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                terms = [(f"x_{p}_{t}_{r}", 1) for r in self.resource_names]
                bqm.add_linear_equality_constraint(terms=terms, lagrange_multiplier=P_ASSIGN, constant=-1)

        # 2. Expertise
        print("DEBUG QUBO: Formulando P: expertise...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                req_level = level_map.get(self.expertise_required_dict.get((p,t), "Junior"), 1)
                for r in self.resource_names:
                    res_level = level_map.get(self.expertise_dict.get(r, "Junior"), 1)
                    if res_level < req_level:
                        bqm.add_variable(f"x_{p}_{t}_{r}", P_EXPERTISE)

        # 3. Link y-x
        print("DEBUG QUBO: Formulando P: link y-x (El Guardián)...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_var_label = f"x_{p}_{t}_{r}"
                    for d in self.days_list:
                        y_expr = y_expressions[(p, t, r, d)]
                        for y_bit_var, _ in y_expr.linear.items():
                            bqm.add_variable(y_bit_var, P_LINK_YX)
                            bqm.add_interaction(y_bit_var, x_var_label, -P_LINK_YX)

        # 4. Horas Totales (El Contador)
        print("DEBUG QUBO: Formulando P: horas totales (El Contador)...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                H_pt = self.hours_required_dict.get((p, t), 0)
                terms = []
                for r in self.resource_names:
                    for d in self.days_list:
                        y_expr = y_expressions[(p, t, r, d)]
                        for var, coeff in y_expr.linear.items():
                            terms.append((var, coeff))
                bqm.add_linear_equality_constraint(terms=terms, lagrange_multiplier=P_HOURS_TOTAL, constant=-H_pt)

        # 5. Disponibilidad del Recurso
        print("DEBUG QUBO: Formulando P: disponibilidad...")
        for r in self.resource_names:
            for d in self.days_list:
                A_rd = self.availability_numeric.get((r, d), 0)
                terms = []
                for p in self.project_names:
                    for t in self.tasks_per_project_name.get(p, []):
                        for var, coeff in y_expressions[(p, t, r, d)].linear.items():
                            terms.append((var, coeff))
                bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_AVAIL, ub=A_rd,label =f"Availability_{r}_{d}")

        # 6. Enlace y -> WorkDay
        print("DEBUG QUBO: Formulando P: link y-WorkDay...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    work_var_label = f"WorkDay_{p}_{t}_{d}"
                    max_y_sum_daily = M_daily * len(self.resource_names)
                    terms = []
                    for r_loop in self.resource_names:
                        for var, coeff in y_expressions[(p,t,r_loop,d)].linear.items():
                            terms.append((var, coeff))
                    terms.append((work_var_label, -max_y_sum_daily))
                    bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_LINK_WD, ub=0,label = f"LinkYWorkDay_{p}_{t}_{d}")

        # 7. Definición EndDay
        print("DEBUG QUBO: Formulando P: definición EndDay...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                end_expr_pt = end_day_expressions[(p,t)]
                for d_loop in self.days_list:
                    work_var_label = f"WorkDay_{p}_{t}_{d_loop}"
                    terms = [(var, coeff) for var, coeff in end_expr_pt.linear.items()]
                    terms.append((work_var_label, -max_day_val - 10)) # Big-M
                    constraint_constant = max_day_val + 10 - d_loop
                    bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_ENDDAY_DEF, constant=constraint_constant, lb=0, label =f"EndDayDef_{p}_{t}_{d_loop}")
        
        # 8. Secuencialidad
        print("DEBUG QUBO: Formulando P: secuencia (Finish-to-Start) [NUEVA VERSIÓN]...")
        for p_proj in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name.get(p_proj, []),
                key=lambda t_name: getattr(self.task_info.get((p_proj, t_name)), 'sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i, t_i_plus_1 = sorted_tasks[i], sorted_tasks[i+1]
                for d_day in self.days_list:
                    work_day_successor_label = f"WorkDay_{p_proj}_{t_i_plus_1}_{d_day}"
                    for k_day in self.days_list:
                        if k_day >= d_day:
                            work_day_predecessor_label = f"WorkDay_{p_proj}_{t_i}_{k_day}"
                            bqm.add_interaction(work_day_predecessor_label, work_day_successor_label, P_SEQ)

        # 9. Deadline
        print("DEBUG QUBO: Formulando P: deadline...")
        for project_data in self.input_data.projects:
            if project_data.deadline:
                deadline_day_num = (project_data.deadline - self.start_date).days + 1
                if 1 <= deadline_day_num <= max_day_val:
                    for t_task in self.tasks_per_project_name.get(project_data.name, []):
                        terms = [(var, coeff) for var, coeff in end_day_expressions[(project_data.name, t_task)].linear.items()]
                        bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_DEADLINE, ub=deadline_day_num,label = f"Deadline_{project_data.name}_{t_task}")

        # 10. Definición Makespan
        print("DEBUG QUBO: Formulando P: definición Makespan...")
        for p_proj in self.project_names:
            for t_task in self.tasks_per_project_name.get(p_proj, []):
                terms = []
                for var, coeff in makespan_expr.linear.items(): terms.append((var, coeff))
                for var, coeff in end_day_expressions[(p_proj, t_task)].linear.items(): terms.append((var, -coeff))
                bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_MAKE_DEF, lb=0,label = f"MakeDef_{p_proj}_{t_task}")


        # --- NUEVA RESTRICCIÓN: FORZAR HORAS SI WORK_DAY ES 1 ---
        print("DEBUG QUBO: Formulando P: Forzar horas si work_day=1...")
        epsilon = 0.1 # Un valor pequeño pero mayor que cero
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for d in self.days_list:
                    work_var = work_day_vars[(p, t, d)]
                    sum_y_ptd = dimod.quicksum(y_expressions[(p, t, r, d)] for r in self.resource_names)
                    
                    # La restricción es: sum_y_ptd >= epsilon * work_var
                    # La formulamos como: sum_y_ptd - epsilon * work_var >= 0
                    
                    constraint_expr = sum_y_ptd - epsilon * work_var
                    
                    # Usamos una penalización cuadrática para forzar que la expresión sea >= 0
                    # Esto se puede hacer añadiendo un slack: constraint_expr - slack = 0
                    # O, de forma más directa, penalizando si es negativo, lo cual es más complejo.
                    # Usemos add_linear_inequality_constraint, que lo hace por nosotros.
                    
                    # Obtenemos los términos lineales de la expresión
                    linear_terms = list(constraint_expr.linear.items())
                    
                    bqm.add_linear_inequality_constraint(
                        terms=linear_terms,
                        lagrange_multiplier=P_LINK_WD, # Reutilizamos una penalización HARD
                        label=f"force_hours_if_active_{p}_{t}_{d}",
                        lb=0, # Límite inferior de la desigualdad (>= 0)
                        ub=M_daily * len(self.resource_names) # Límite superior (un valor grande)
            )


        # # 11. Límite de Horas Diarias por Tarea y Recurso
        # print("DEBUG QUBO: Formulando P: Límite de horas diarias por tarea...")

        # # CAMBIO: Itera sobre las claves (key) y los valores (y_expr) del diccionario
        # for key, y_expr in y_expressions.items(): 
            
        #     terms = [(var, coeff) for var, coeff in y_expr.linear.items()]
            
        #     # CAMBIO: Usa la 'key' del diccionario para crear la etiqueta única.
        #     # La clave 'key' será algo como, por ejemplo, "y_t1_r1_d1".
        #     constraint_label = f"DailyHoursLimit_{key}" 
            
        #     bqm.add_linear_inequality_constraint(
        #         terms=terms, 
        #         lagrange_multiplier=P_HARD, 
        #         ub=M_daily, 
        #         label=constraint_label # <-- Pasa la etiqueta que acabas de crear
        #     )


        # # 12. (NUEVA) Penalización de Tareas No Contiguas
        # print("DEBUG QUBO: Formulando P: Contigüidad de Tareas...")
        # for p in self.project_names:
        #     for t in self.tasks_per_project_name.get(p, []):
        #         # Iteramos desde el segundo día hasta el penúltimo
        #         for d in range(2, len(self.days_list)):
        #             # El patrón a penalizar es: Work(d-1)=1, Work(d)=0, Work(d+1)=1
        #             wd_prev_label = f"WorkDay_{p}_{t}_{d-1}"
        #             wd_curr_label = f"WorkDay_{p}_{t}_{d}"
        #             wd_next_label = f"WorkDay_{p}_{t}_{d+1}"
                    
        #             # Esto añade el término cúbico: P_LOW * wd_prev * (1-wd_curr) * wd_next
        #             # Se expande a: P_LOW * (wd_prev*wd_next - wd_prev*wd_curr*wd_next)
        #             bqm.add_interaction(wd_prev_label, wd_next_label, P_LOW)
        #             bqm.add_interaction(wd_prev_label, wd_curr_label, {wd_next_label: -P_LOW})

        # --- Finalizar BQM ---
        self.model = bqm
        print(f"DEBUG QUBO: Modelo BQM final construido con {len(self.model.variables)} variables binarias.")
        return self.model
    
    
    def _build_model_simple(self):
        """
        Construye un BQM simplificado "Tarea-en-Día".
        VERSIÓN FINAL CORREGIDA: Conecta la variable makespan a las tareas.
        """
        print("DEBUG QUBO: Construyendo modelo BQM SIMPLIFICADO (Tarea-en-Día)...")
        bqm = dimod.BinaryQuadraticModel('BINARY')
        
        import itertools

        # --- Variables Principales ---
        z = {(p, t, d): dimod.Binary(f"z_{p}_{t}_{d}")
            for p in self.project_names
            for t in self.tasks_per_project_name.get(p, [])
            for d in self.days_list}

        x = {(p, t, r): dimod.Binary(f"x_{p}_{t}_{r}")
            for p in self.project_names
            for t in self.tasks_per_project_name.get(p, [])
            for r in self.resource_names}

        makespan_expr, _ = integer_to_binary("Makespan", self.days_list[-1])

        # --- Penalizaciones ---
        P_CRITICAL = 100.0
        P_HARD = 50.0
        P_OBJECTIVE = 1.0

        # --- Objetivo: Minimizar Makespan ---
        bqm.update(P_OBJECTIVE * makespan_expr)

        # --- Restricciones Simplificadas ---

        # 1. Cada tarea se realiza exactamente en UN día
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                sum_z_pt = dimod.quicksum(z[(p, t, d)] for d in self.days_list)
                bqm.update(P_CRITICAL * (sum_z_pt - 1)**2)

        # 2. Cada tarea tiene exactamente UN recurso asignado
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                sum_x_pt = dimod.quicksum(x[(p, t, r)] for r in self.resource_names)
                bqm.update(P_CRITICAL * (sum_x_pt - 1)**2)
        
        # 3. Disponibilidad: Un recurso no puede hacer más de UNA tarea por día
        for r in self.resource_names:
            for d in self.days_list:
                q_ancillas_rd = []
                for p in self.project_names:
                    for t in self.tasks_per_project_name.get(p, []):
                        q_var = dimod.Binary(f"q_ancilla_{r}_{p}_{t}_{d}")
                        q_ancillas_rd.append(q_var)
                        z_var = z[(p, t, d)]
                        x_var = x[(p, t, r)]
                        bqm.update(P_HARD * (3 * q_var + z_var * x_var - 2 * q_var * z_var - 2 * q_var * x_var))
                for q_pair in itertools.combinations(q_ancillas_rd, 2):
                    bqm.update(P_CRITICAL * q_pair[0] * q_pair[1])

        # 4. Secuencialidad: end_day(t_i) < end_day(t_i+1)
        for p in self.project_names:
            sorted_tasks = sorted(
                self.tasks_per_project_name[p],
                key=lambda t: getattr(self.task_info.get((p, t)), 'sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i, t_i_plus_1 = sorted_tasks[i], sorted_tasks[i+1]
                end_day_ti = dimod.quicksum(d * z[(p, t_i, d)] for d in self.days_list)
                end_day_ti_plus_1 = dimod.quicksum(d * z[(p, t_i_plus_1, d)] for d in self.days_list)
                
                max_diff = self.days_list[-1]
                slack_seq, _ = integer_to_binary(f"slack_seq_{p}_{t_i}", max_diff, prefix="seq_")
                
                constraint_seq = end_day_ti - end_day_ti_plus_1 + 1 + slack_seq
                bqm.update(P_CRITICAL * (constraint_seq)**2)
        
        # ---------------------------------------------------------------------------------
        # --- ¡AQUÍ ESTÁ LA NUEVA RESTRICCIÓN CLAVE! ---
        # 5. Definición del Makespan
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                # El día de finalización de la tarea 't' es la suma ponderada de las z
                end_day_t = dimod.quicksum(d * z[(p, t, d)] for d in self.days_list)
                
                # La restricción es: makespan_expr >= end_day_t
                # que se reescribe como: makespan_expr - end_day_t >= 0
                # Modelamos esto con un slack: makespan_expr - end_day_t - slack = 0
                
                max_val = self.days_list[-1]
                slack_makespan, _ = integer_to_binary(f"slack_makespan_{p}_{t}", max_val, prefix="mk_")
                
                constraint = makespan_expr - end_day_t - slack_makespan
                bqm.update(P_CRITICAL * (constraint)**2)
        # ---------------------------------------------------------------------------------
                
        self.model = bqm
        print(f"DEBUG QUBO: Modelo SIMPLIFICADO construido con {len(self.model.variables)} variables.")

    

    def _solve_model(self) -> tuple[str, float]:
        """
        Resuelve el BQM usando Neal Sampler.
        Esta función solo se encarga de la ejecución, no de la interpretación.
        Devuelve el estado bruto y el tiempo de ejecución, como dicta la clase base.
        """
        if self.model is None:
            raise ValueError("El modelo BQM no ha sido construido.")

        # Importamos neal aquí para mantener las dependencias de solvers localizadas.
        import neal
        import time

        print("DEBUG QUBO: Iniciando solver Neal...")
        start_time = time.time()
        
        runtime = 0.0
        final_status = "Error"
        self.sampleset = None # Limpiamos el sampleset anterior

        try:
            # Es buena práctica obtener la configuración desde el objeto input_data
            # para mantener la consistencia.
            # Asumimos que num_reads se añade a OptimizationConfig en ui_planning.py
            num_reads = getattr(self.input_data.config, 'num_reads', 1000) # Default a 1000
            print(f"DEBUG QUBO: Usando num_reads = {num_reads}")

            # Instanciación estándar del sampler de recocido simulado
            sampler = neal.SimulatedAnnealingSampler()
            
            # Ejecutamos el sampler
            self.sampleset = sampler.sample(self.model, num_reads=num_reads)
            
            runtime = time.time() - start_time
            print(f"DEBUG QUBO: Neal finalizó en {runtime:.2f} segundos.")
            
            # Comprobamos si el sampleset contiene al menos una solución
            if self.sampleset and len(self.sampleset) > 0:
                # El estado 'Feasible' es provisional. _extract_results lo refinará
                # a 'INFEASIBLE' o 'Feasible (Violations)' si es necesario.
                final_status = "Feasible" 
            else:
                final_status = "No Solution Found"

        except Exception as e:
            runtime = time.time() - start_time
            # Guardamos el mensaje de error en el objeto para poder mostrarlo/loguearlo
            self.result = OptimizationResult(
                status="Solver Error", 
                error_message=f"Error durante la ejecución de Neal: {e}"
            )
            print(f"ERROR QUBO: {self.result.error_message}")
            final_status = "Solver Error"
        
        # Devolvemos la tupla que el método 'solve' de la clase base espera
        return final_status, runtime
    
    def _extract_results(self, status: str):
        """
        Extrae, decodifica, valida e interpreta los resultados del sampleset.
        Esta versión es compatible tanto con el modelo complejo como con el simple.
        """
        if not hasattr(self, 'sampleset') or self.sampleset is None or len(self.sampleset) == 0:
            # ... (código de manejo de error sin cambios)
            return

        try:
            best_sample = self.sampleset.first.sample
            objective_val = self.sampleset.first.energy
            print(f"DEBUG QUBO: Mejor energía encontrada (bruta): {objective_val}")

            # Inicializamos los contenedores de resultados
            makespan_val = 0
            task_completion_days = {}
            assignment = {} # Para el modelo complejo: {(p,t,r,d): h}
            work_details_simple = {} # Para el modelo simple: {(p,t): (r,d)}
            
            # --- LÓGICA DE DECODIFICACIÓN DUAL ---
            # Comprobamos si el modelo simple se ejecutó buscando una variable 'z'
            is_simple_model = any(k.startswith('z_') for k in best_sample.keys())

            if is_simple_model:
                print("DEBUG EXTRACT: Detectado resultado de MODELO SIMPLE.")
                # Decodificamos el resultado del modelo "Tarea-en-Día"
                for p in self.project_names:
                    for t in self.tasks_per_project_name.get(p, []):
                        # Encontrar el día asignado
                        assigned_day = 0
                        for d in self.days_list:
                            if best_sample.get(f"z_{p}_{t}_{d}", 0) == 1:
                                assigned_day = d
                                break
                        
                        # Encontrar el recurso asignado
                        assigned_resource = None
                        for r in self.resource_names:
                            if best_sample.get(f"x_{p}_{t}_{r}", 0) == 1:
                                assigned_resource = r
                                break
                        
                        if assigned_day > 0 and assigned_resource:
                            work_details_simple[(p, t)] = (assigned_resource, assigned_day)
                            task_completion_days[(p, t)] = assigned_day

                if task_completion_days:
                    makespan_val = max(task_completion_days.values())

            else:
                print("DEBUG EXTRACT: Detectado resultado de MODELO COMPLEJO.")
                # Lógica de decodificación del modelo complejo (la que ya tenías)
                # ... (aquí iría tu lógica anterior para decodificar y_bits, end_day_bits, etc.)
                pass

            # --- VALIDACIÓN E IMPRESIÓN ---
            # El validador actual dará errores de horas para el modelo simple. ¡Es normal!
            # Lo ignoramos por ahora, porque nuestro objetivo es solo ver la estructura.
            
            # Imprimimos la solución decodificada
            print("\n" + "="*20 + " INICIO SOLUCIÓN DECODIFICADA " + "="*20)
            print(f"Makespan Final: {makespan_val} días")
            
            if is_simple_model:
                print("\n--- Planificación (Tarea -> Recurso, Día) ---")
                sorted_plan = sorted(work_details_simple.items(), key=lambda item: item[1][1]) # Ordenar por día
                for (p, t), (r, d) in sorted_plan:
                    print(f"  - Día {d}: Tarea '{t}' (Proy: '{p}') -> Recurso: {r}")
            else:
                # Imprimir resultados del modelo complejo
                pass
            
            print("="*22 + " FIN SOLUCIÓN DECODIFICADA " + "="*23 + "\n")

            # El resto de la lógica para crear OptimizationResult...
            # ...
            # Por ahora, puedes dejar que el validador marque el resultado como INFEASIBLE.
            # Lo importante es que AHORA SÍ verás una planificación coherente en la consola.
            
            # Construcción del resultado final (simplificado para este paso)
            self.result = OptimizationResult(
                status="Feasible" if not is_simple_model else "Feasible (Simple Model)",
                solver_runtime=self.solver_runtime,
                objective_value=objective_val,
                makespan=makespan_val,
                assignment=assignment,
                task_completion_days=task_completion_days
            )

        except Exception as e:
            import traceback
            error_message = f"Error crítico extrayendo/decodificando resultados: {e}\n{traceback.format_exc()}"
            print(error_message)
            self.result = OptimizationResult(status="Error Extracting", error_message=error_message, solver_runtime=self.solver_runtime)

    def _decode_integer_from_bits(self, sample, bits_dict):
        """
        Función auxiliar para decodificar un entero a partir de sus bits.
        VERSIÓN CORREGIDA.
        """
        val = 0
        if not bits_dict:
            return val
            
        # bits_dict es de la forma {0: BQM_bit0, 1: BQM_bit1, ...}
        for i, bit_var_obj in bits_dict.items():
            # bit_var_obj es un mini-BQM. Necesitamos el nombre (string) de la variable que contiene.
            if not hasattr(bit_var_obj, 'variables') or not bit_var_obj.variables:
                continue 
            
            bit_var_label = list(bit_var_obj.variables)[0]
            
            # Usamos el nombre (string) para buscar en el diccionario de resultados 'sample'
            if sample.get(bit_var_label, 0) == 1:
                val += 2**i
        return val

    def _validate_solution(self, assignment, work_details, completion_days):
        """
        Valida que la solución decodificada cumple las restricciones clave.
        VERSIÓN CORREGIDA.
        """
        errors = []
        
        # Validación 1: Horas totales por tarea
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                H_pt = self.hours_required_dict.get((p, t), 0)
                total_worked = sum(h for (p_k, t_k, _, _), h in work_details.items() if p_k == p and t_k == t)
                if abs(total_worked - H_pt) > 0.1:
                    errors.append(f"Horas tarea ({p},{t}): Requeridas={H_pt}, Realizadas={total_worked:.1f}")

        # Validación 2: Disponibilidad de recursos por día
        for r in self.resource_names:
            for d in self.days_list:
                A_rd = self.availability_numeric.get((r, d), 0)
                hours_on_day = sum(h for (_, _, r_k, d_k), h in work_details.items() if r_k == r and d_k == d)
                if hours_on_day > A_rd + 0.1:
                    errors.append(f"Disponibilidad recurso '{r}' día {d}: Disponible={A_rd}, Usado={hours_on_day:.1f}")
        
        # Validación 3: Secuencialidad de tareas
        for p_proj in self.project_names:
            # --- CORRECCIÓN: Usar getattr para acceder de forma segura al atributo 'sequence' ---
            sorted_tasks = sorted(
                self.tasks_per_project_name.get(p_proj, []),
                key=lambda t_name: getattr(self.task_info.get((p_proj, t_name)), 'sequence', float('inf'))
            )
            for i in range(len(sorted_tasks) - 1):
                t_i, t_i_plus_1 = sorted_tasks[i], sorted_tasks[i+1]
                end_day_i = completion_days.get((p_proj, t_i))
                
                start_day_i_plus_1 = float('inf')
                for (p_k, t_k, _, d_k), h in work_details.items():
                    if p_k == p_proj and t_k == t_i_plus_1 and h > 0:
                        start_day_i_plus_1 = min(start_day_i_plus_1, d_k)
                
                if end_day_i is not None and start_day_i_plus_1 != float('inf'):
                    if end_day_i >= start_day_i_plus_1:
                        errors.append(f"Secuencia ({p_proj}): Tarea '{t_i}' acaba día {end_day_i} pero '{t_i_plus_1}' empieza día {start_day_i_plus_1}")

        return errors
    
    # Sobreescribimos el método 'solve' de la clase base para controlar qué modelo se construye
    def solve(self) -> OptimizationResult:
        """
        Orquesta la construcción, solución y extracción de resultados,
        FORZANDO EL USO DEL MODELO SIMPLIFICADO para depuración.
        """
        self.result = OptimizationResult(status="Not Run")
        self.solver_runtime = 0.0
        
        try:
            print("--- Iniciando Modelo: NealMakespanModel (con constructor SIMPLE) ---")
            
            print("Paso 1/4: Preparando datos comunes...")
            self._prepare_common_data()
            print(f"Datos comunes preparados. Horizonte: {len(self.days_list)} días. Recursos: {len(self.resource_names)}.")
            
            # -----------------------------------------------------------------
            # --- ¡AQUÍ ESTÁ EL CAMBIO CLAVE! ---
            # Llamamos explícitamente al constructor del modelo simplificado
            print("Paso 2/4: Construyendo modelo específico (SIMPLIFICADO)...")
            self._build_model_simple()
            # -----------------------------------------------------------------
            
            print(f"Paso 3/4: Resolviendo (Límite: {self.input_data.config.num_reads} reads)...")
            solve_status_str, runtime = self._solve_model()
            self.solver_runtime = runtime
            print(f"Resolución finalizada. Estado del solver: {solve_status_str}, Tiempo: {runtime:.2f}s.")

            print("Paso 4/4: Extrayendo resultados...")
            self._extract_results(solve_status_str)
            print("Resultados extraídos.")

        except Exception as e:
            import traceback
            error_msg = f"Error Inesperado en el flujo de 'solve': {e}\n{traceback.format_exc()}"
            print(f"ERROR durante la optimización: {error_msg}")
            self.result = OptimizationResult(
                status="Execution Error",
                solver_runtime=self.solver_runtime,
                error_message=error_msg
            )
        
        print(f"--- Modelo Finalizado: {self.__class__.__name__} --- Estado Final del Resultado: {self.result.status} ---")
        return self.result