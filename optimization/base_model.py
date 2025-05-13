# optimization/base_model.py
from abc import ABC, abstractmethod
from optimization.data_models import OptimizationInput, OptimizationResult # Importar desde el archivo de modelos de datos
import time
from datetime import timedelta

class OptimizationModel(ABC):
    """Clase base abstracta para los modelos de optimización."""

    def __init__(self, input_data: OptimizationInput):
        self.input_data = input_data
        self.model = None  # Objeto del modelo (PuLP o PySCIPOpt)
        self.variables = {} # Diccionario para guardar variables del modelo
        self.result = OptimizationResult(status="Not Run") # Resultado inicial
        self.solver_runtime = 0.0

    def _prepare_common_data(self):
        """Prepara datos comunes usados por varios modelos (ej: mapeos)."""
        # Ejemplo: Crear mapeos o diccionarios básicos desde las listas de objetos
        self.project_dict = {p.id: p for p in self.input_data.projects}
        self.task_dict = {t.id: t for t in self.input_data.tasks}
        self.resource_dict = {r.name: r for r in self.input_data.resources}

        self.project_names = [p.name for p in self.input_data.projects]
        self.resource_names = [r.name for r in self.input_data.resources]

        self.expertise_dict = {r.name: r.expertise for r in self.input_data.resources}
        self.cost_dict = {r.name: r.cost for r in self.input_data.resources}

        # Procesar tareas por proyecto
        self.tasks_per_project_name = {p.name: [] for p in self.input_data.projects}
        self.hours_required_dict = {}
        self.expertise_required_dict = {}
        self.task_info = {} # Para secuencia, etc.
        for task in self.input_data.tasks:
             project = self.project_dict.get(task.project_id)
             if project:
                 proj_name = project.name
                 task_name = task.name
                 task_key = (proj_name, task_name)
                 # Evitar duplicados por nombre si es necesario
                 if task_name not in self.tasks_per_project_name[proj_name]:
                      self.tasks_per_project_name[proj_name].append(task_name)
                      self.hours_required_dict[task_key] = task.hours
                      self.expertise_required_dict[task_key] = task.expertise
                      self.task_info[task_key] = task # Guardar objeto Task

        # Calcular horizonte y disponibilidad numérica
        self.start_date = self.input_data.config.start_date
        # (Lógica para calcular n_days y days_list como en ui_planning.py)
        # ...
        # self.days_list = list(range(1, n_days + 1))
        # self.M_days = n_days + 1
        # ...
        # (Lógica para calcular availability_numeric como en ui_planning.py)
        # ...
        # self.availability_numeric = availability_numeric
        # ...

        # Calcular horizonte y disponibilidad numérica (¡Implementar lógica!)
        # Esta lógica es compleja y depende de cómo calcules el horizonte
        # y mapees los días. Deberías mover la lógica de prepare_solver_data aquí.
        # Placeholder:
        self.days_list = list(range(1, self.input_data.config.planning_horizon_days + 1)) if self.input_data.config.planning_horizon_days else list(range(1, 91)) # Default 90 days
        self.M_days = len(self.days_list) + 1
        # Placeholder para disponibilidad - ¡NECESITA IMPLEMENTACIÓN COMPLETA!
        self.availability_numeric = {(r.name, d): 8 for r in self.input_data.resources for d in self.days_list if (self.start_date + timedelta(days=d-1)).weekday() < 5}


    @abstractmethod
    def _build_model(self):
        """Define variables, objetivo y restricciones específicas."""
        pass

    @abstractmethod
    def _solve_model(self) -> tuple[str, float]:
        """Llama al solver apropiado y devuelve (status_str, runtime)."""
        pass

    @abstractmethod
    def _extract_results(self, status: str):
        """Extrae los resultados del modelo resuelto y los guarda en self.result."""
        pass

    def solve(self) -> OptimizationResult:
        """Orquesta la construcción, solución y extracción de resultados."""
        status = "Error"
        try:
            print(f"--- Iniciando Modelo: {self.__class__.__name__} ---")
            print("Preparando datos comunes...")
            self._prepare_common_data() # Llama a la preparación de datos
            print("Construyendo modelo...")
            self._build_model() # Llama a la construcción específica
            print(f"Resolviendo con límite de tiempo: {self.input_data.config.solver_time_limit}s...")
            status, runtime = self._solve_model() # Llama a la solución específica
            self.solver_runtime = runtime
            print(f"Extrayendo resultados (Status: {status})...")
            self._extract_results(status) # Llama a la extracción específica
            print(f"--- Modelo Finalizado ---")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR durante la optimización: {e}\n{error_msg}")
            self.result = OptimizationResult(
                status="Error",
                solver_runtime=self.solver_runtime, # Puede tener tiempo acumulado
                error_message=error_msg
            )
        return self.result