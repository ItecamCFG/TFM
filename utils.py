### utils.py
#### Este script contiene funciones auxiliares para la optimización de tareas y recursos.

import csv
from datetime import datetime
import os

def save_solver_result(result, input_data, dataset_name="unknown_dataset"):
    """
    Guarda una fila en un CSV con los resultados del solver.
    """
    output_file = "solver_benchmark_results.csv"
    row = {
        "dataset_name": dataset_name,
        "projects": len(input_data.projects),
        "tasks": len(input_data.tasks),
        "resources": len(input_data.resources),
        "status": result.status,
        "makespan": result.makespan if result.makespan is not None else "",
        "solver_runtime": round(result.solver_runtime, 2),
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
