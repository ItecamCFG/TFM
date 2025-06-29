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
        """
        Versión de DEPURACIÓN del modelo simplificado.
        Objetivo: Aislar y verificar el mecanismo de asignación de trabajo.
        """
        print("DEBUG QUBO: Construyendo modelo en MODO DEPURACIÓN...")
        
        # --- 1. PREPARACIÓN DE CONSTANTES ---
        level_map = {"Junior": 1, "Senior": 2, "Experto": 3}
        BLOCK_SIZE = 8
        max_day_val = self.days_list[-1] if self.days_list else 1
        
        bqm = dimod.BinaryQuadraticModel('BINARY')
        makespan_expr, _ = integer_to_binary("Makespan", max_day_val)

        # --- 2. COEFICIENTES DE PENALIZACIÓN (SIMPLIFICADOS PARA DEPURAR) ---
        # Usaremos valores absolutos y claros para entender su efecto.
        # La penalización por no hacer el trabajo debe ser mayor que cualquier posible
        # beneficio de tener un makespan bajo.
        P_CUMPLIR_HORAS = 500.0  # ¡Una multa muy alta por no trabajar!
        P_ASIGNACION = 200.0     # Multa por errores de asignación.
        P_SECUENCIA = 200.0      # Multa por romper la secuencia.
        P_DISPONIBILIDAD = 100.0 # Multa por sobrecarga.
        P_OBJETIVO = 1.0         # El "premio" por un makespan bajo es relativamente pequeño.

        self.penalties = { 'P_CUMPLIR_HORAS': P_CUMPLIR_HORAS, 'P_ASIGNACION': P_ASIGNACION }
            
        # --- 3. OBJETIVO ---
        # bqm.update(P_OBJECTIVE * makespan_expr) # <-- Desactivamos temporalmente el objetivo

        # --- 4. RESTRICCIONES (EN MODO DEPURACIÓN) ---

        # R1: Asignación Única (ESENCIAL)
        print("DEBUG QUBO: Formulando P: Asignación Única...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                terms = [(f"x_{p}_{t}_{r}", 1) for r in self.resource_names]
                bqm.add_linear_equality_constraint(terms=terms, lagrange_multiplier=P_ASIGNACION, constant=-1)

        # R2: Expertise (DESACTIVADA TEMPORALMENTE)
        # print("DEBUG QUBO: Formulando P: Expertise...")

        # R3: Vínculo Asignación-Trabajo (x -> z) (ESENCIAL)
        print("DEBUG QUBO: Formulando P: Vínculo Asignación-Trabajo...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                for r in self.resource_names:
                    x_label = f"x_{p}_{t}_{r}"
                    for d in self.days_list:
                        z_label = f"z_{r}_{t}_{d}"
                        terms = [(z_label, 1), (x_label, -1)]
                        bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_ASIGNACION, label=f"link_xz_{r}_{t}_{d}", ub=0)

        # R4: Horas Totales (ESENCIAL)
        print("DEBUG QUBO: Formulando P: Horas Totales...")
        for p in self.project_names:
            for t in self.tasks_per_project_name.get(p, []):
                H_pt = self.hours_required_dict.get((p, t), 0)
                terms = [(f"z_{r}_{t}_{d}", -BLOCK_SIZE) for r in self.resource_names for d in self.days_list]
                bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=P_CUMPLIR_HORAS, label=f"total_hours_{p}_{t}", ub=-H_pt)

        # --- RESTRICCIONES DESACTIVADAS ---
        # R5: Disponibilidad (DESACTIVADA TEMPORALMENTE)
        # R6: Horas por Día (DESACTIVADA TEMPORALMENTE)
        # R7: Secuencialidad (DESACTIVADA TEMPORALMENTE)
        
        self.model = bqm
        print(f"DEBUG QUBO: Modelo de DEPURACIÓN construido con {len(self.model.variables)} variables.")
        return self.model

    def _solve_model(self) -> tuple[str, float]:
        """
        Resuelve el BQM usando Neal Sampler y devuelve (status, runtime).
        ESTA ES LA VERSIÓN CORREGIDA.
        """
        if self.model is None:
            raise ValueError("El modelo BQM no ha sido construido.")

        print("DEBUG QUBO: Iniciando solver Neal...")
        start_time = time.time()
        
        runtime = 0.0
        final_status = "Error"

        try:
            num_reads = self.input_data.config.num_reads
            print(f"DEBUG QUBO: Usando num_reads = {num_reads}")

            sampler = Neal()
            self.sampleset = sampler.sample(self.model, num_reads=num_reads)
            
            runtime = time.time() - start_time
            print(f"DEBUG QUBO: Neal finalizó en {runtime:.2f} segundos.")
            
            if self.sampleset:
                final_status = "Feasible" # El estado se refinará en _extract_results
            else:
                final_status = "No Solution Found"

        except Exception as e:
            runtime = time.time() - start_time
            self.error_message = f"Error durante la ejecución del solver Neal: {e}"
            print(f"ERROR QUBO: {self.error_message}")
            final_status = "Error"
        
        # Devuelve la tupla que el contrato exige: (string, float)
        return final_status, runtime

    def _extract_results(self, status: str):
        """
        Extrae, decodifica, valida e interpreta los resultados del sampleset.
        VERSIÓN CORREGIDA.
        """
        if not hasattr(self, 'sampleset') or self.sampleset is None or len(self.sampleset) == 0:
            self.result = OptimizationResult(
                status="No Solution Found" if status != "Error" else "Error",
                solver_runtime=self.solver_runtime,
                error_message="El solver no devolvió un sampleset."
            )
            return

        try:
            best_sample = self.sampleset.first.sample
            objective_val = self.sampleset.first.energy
            print(f"DEBUG QUBO: Mejor energía encontrada (bruta): {objective_val}")

            # --- 1. DECODIFICACIÓN DE VARIABLES ---
            
            # Decodificar Makespan, EndDay, y Horas (usan la función auxiliar corregida)
            makespan_val = self._decode_integer_from_bits(best_sample, self.variables.get('makespan_bits', {}))
            
            task_completion_days = {
                (p, t): self._decode_integer_from_bits(best_sample, bits_dict)
                for (p, t), bits_dict in self.variables.get('end_day_bits', {}).items()
            }
            
            work_details = {}
            for (p, t, r, d), bits_dict in self.variables.get('y_bits', {}).items():
                hours_val = self._decode_integer_from_bits(best_sample, bits_dict)
                if hours_val > 0.1: # Usar tolerancia
                    work_details[(p, t, r, d)] = hours_val

            # --- CORRECCIÓN: Decodificar 'x' reconstruyendo los nombres (labels) ---
            # Es la forma más robusta, no depende de haber guardado las variables.
            task_assignment = {}
            for p in self.project_names:
                for t in self.tasks_per_project_name.get(p, []):
                    for r in self.resource_names:
                        # Reconstruimos el nombre exacto de la variable 'x'
                        x_var_label = f"x_{p}_{t}_{r}"
                        # Buscamos ese nombre en los resultados del solver
                        if best_sample.get(x_var_label, 0) == 1:
                            task_assignment[(p, t)] = r

            # --- INICIO DEL BLOQUE DE DEPURACIÓN DE LA SOLUCIÓN ---
            print("\n" + "="*20 + " INICIO SOLUCIÓN DECODIFICADA " + "="*20)
            print(f"Makespan Final: {makespan_val} días")
            
            print("\n--- Asignación de Tareas (Tarea -> Recurso) ---")
            if task_assignment:
                for (p, t), r in task_assignment.items():
                    print(f"  - Tarea '{t}' (Proy: '{p}') -> Recurso: {r}")
            else:
                print("  - Ninguna tarea fue asignada.")

            print("\n--- Desglose de Horas Trabajadas (Tarea, Recurso, Día -> Horas) ---")
            if work_details:
                # Ordenamos los detalles para una lectura más fácil
                sorted_work = sorted(work_details.items(), key=lambda item: (item[0][3], item[0][0], item[0][1]))
                for (p, t, r, d), h in sorted_work:
                    print(f"  - Día {d}: Tarea '{t}' (Rec: {r}) -> {h} horas")
            else:
                print("  - No se registraron horas de trabajo.")
            
            print("\n--- Días de Finalización por Tarea ---")
            if task_completion_days:
                for (p, t), end_d in task_completion_days.items():
                    print(f"  - Tarea '{t}' (Proy: '{p}') finaliza el día: {end_d}")
            else:
                print("  - No se calcularon días de finalización.")

            print("="*22 + " FIN SOLUCIÓN DECODIFICADA " + "="*23 + "\n")
            # --- FIN DEL BLOQUE DE DEPURACIÓN ---
            
            # --- 2. VALIDACIÓN DE LA SOLUCIÓN DECODIFICADA ---
            validation_errors = self._validate_solution(task_assignment, work_details, task_completion_days)

            # --- 3. DETERMINAR ESTADO FINAL Y PREPARAR RESULTADO ---
            final_status = status
            error_message = None
            if validation_errors:
                final_status = "INFEASIBLE"
                error_message = "La solución de menor energía viola las restricciones:\n" + "\n".join(validation_errors)
                print(f"ADVERTENCIA QUBO: {error_message}")
            
            total_cost = None
            if not validation_errors and hasattr(self, 'cost_dict'):
                total_cost = sum(self.cost_dict.get(r, 0.0) * h for (_, _, r, _), h in work_details.items())

            self.result = OptimizationResult(
                status=final_status,
                solver_runtime=self.solver_runtime,
                objective_value=objective_val,
                makespan=makespan_val,
                total_cost=total_cost,
                assignment=task_assignment,
                work_details=work_details,
                task_completion_days=task_completion_days,
                error_message=error_message
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