### utils.py
#### Este script contiene funciones auxiliares para la optimización de tareas y recursos.

import csv
import os
from datetime import datetime

# Importar las clases de los modelos para poder identificarlos
from optimization.neal_model import NealMakespanModel
from optimization.model_pulp import MakespanMinimizationPuLP
from optimization.model_SCIP import MakespanMinimizationSCIP

def log_experiment(model_instance):
    """
    Guarda una fila en un CSV con los resultados detallados de un experimento.
    Captura parámetros de entrada, configuración del solver y resultados.
    """
    log_filepath = "solver_benchmark_results.csv"  # Usamos el nombre de tu fichero existente
    
    # --- 1. Extraer Datos Comunes ---
    input_data = model_instance.input_data
    result = model_instance.result
    
    # Generar el nombre del problema dinámicamente como pediste
    horizon = len(model_instance.days_list) if hasattr(model_instance, 'days_list') and model_instance.days_list else 0
    problem_name = f"RCPSP_{len(input_data.projects)}p_{len(input_data.tasks)}t_{len(input_data.resources)}r_{horizon}d"

    base_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "problem_name": problem_name,
        "model_class": model_instance.__class__.__name__,
        "status": result.status,
        "makespan": result.makespan,
        "solver_runtime_sec": round(result.solver_runtime, 2),
    }

    # --- 2. Extraer Datos Específicos del Solver ---
    solver_specific_data = {}
    if isinstance(model_instance, NealMakespanModel):
        # Datos específicos de nuestro modelo QUBO
        solver_specific_data = {
            'num_reads': getattr(input_data.config, 'num_reads', None),
            'bqm_variables': len(model_instance.model.variables) if model_instance.model else 0,
            'bqm_energy': result.objective_value,
            'num_validation_errors': len(result.error_message.split('\n')) if result.status == "INFEASIBLE" and result.error_message else 0,
            'validation_errors': result.error_message if result.status == "INFEASIBLE" else ""
        }
        # Añadimos las penalizaciones que guardamos en el modelo
        penalties = getattr(model_instance, 'penalties', {})
        solver_specific_data.update(penalties)

    elif isinstance(model_instance, (MakespanMinimizationPuLP, MakespanMinimizationSCIP)):
        # Datos específicos de los solvers MILP
        solver_specific_data = {
            'solver_time_limit': getattr(input_data.config, 'solver_time_limit', None)
        }
        # Puedes añadir más datos si tus otros modelos los exponen
        
    # --- 3. Combinar y Escribir en el CSV ---
    
    # Unimos los diccionarios
    full_row_data = {**base_data, **solver_specific_data}

    # Definimos TODOS los posibles encabezados. DictWriter manejará las columnas que no apliquen.
    fieldnames = [
        'timestamp', 'problem_name', 'model_class', 'status', 'makespan', 
        'solver_runtime_sec', 'num_reads', 'solver_time_limit', 
        'bqm_variables', 'bqm_energy', 'num_validation_errors', 'validation_errors',
        'P_CRITICAL', 'P_HARD', 'P_MEDIUM', 'P_SEQ', 'P_AVAIL', 'P_HOURS_TOTAL','P_LOW'
    ]

    file_exists = os.path.isfile(log_filepath)
    try:
        with open(log_filepath, "a", newline="", encoding="utf-8") as csvfile:
            # Usamos DictWriter y le pasamos todos los posibles fieldnames
            # restval='' hace que si un dato no existe, escriba una celda vacía
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(full_row_data)
        print(f"DEBUG LOG: Resultados del experimento guardados en {log_filepath}")
    except Exception as e:
        print(f"ERROR LOG: No se pudo guardar el log del experimento. Error: {e}")
