# optimization/base_model.py
from abc import ABC, abstractmethod
from optimization.data_models import OptimizationInput, OptimizationResult # Ajusta la ruta si es necesario
import time
from datetime import date, timedelta # Importar date
import traceback

# Definir los nombres de los días laborables que usa tu aplicación
# Esto debería ser consistente con cómo se almacenan en Resource.availability
WORK_DAYS_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

class OptimizationModel(ABC):
    """Clase base abstracta para los modelos de optimización."""

    def __init__(self, input_data: OptimizationInput):
        self.input_data = input_data
        self.model = None  # Objeto del modelo (PuLP, PySCIPOpt, BQM)
        self.variables = {} # Diccionario para guardar variables del modelo
        self.result = OptimizationResult(status="Not Run") # Resultado inicial
        self.solver_runtime = 0.0

        # Atributos que se poblarán en _prepare_common_data
        self.project_dict: dict = {}
        self.task_dict: dict = {}
        self.resource_dict: dict = {}
        self.project_names: list = []
        self.resource_names: list = []
        self.expertise_dict: dict = {}
        self.cost_dict: dict = {}
        self.tasks_per_project_name: dict = {}
        self.hours_required_dict: dict = {}
        self.expertise_required_dict: dict = {}
        self.task_info: dict = {} # { (proj_name, task_name): Task_object }
        self.start_date: date = date.today()
        self.days_list: list = [] # Lista de números de día [1, 2, ..., N]
        self.availability_numeric: dict = {} # { (resource_name, day_num): hours }
        self.M_days_big_m: int = 0 # Para Big-M en restricciones de días

    def _prepare_common_data(self):
        """Prepara datos comunes usados por varios modelos (ej: mapeos, horizonte, disponibilidad)."""
        print("DEBUG BASE: Iniciando _prepare_common_data...")

        # 1. Mapeos básicos de objetos
        self.project_dict = {p.id: p for p in self.input_data.projects}
        self.task_dict = {t.id_task: t for t in self.input_data.tasks} # Asume que Task tiene un 'id'
        self.resource_dict = {r.name: r for r in self.input_data.resources}

        self.project_names = [p.name for p in self.input_data.projects]
        self.resource_names = [r.name for r in self.input_data.resources]

        self.expertise_dict = {r.name: r.expertise for r in self.input_data.resources}
        self.cost_dict = {r.name: r.cost for r in self.input_data.resources}

        # 2. Procesar tareas por proyecto y extraer info
        self.tasks_per_project_name = {p.name: [] for p in self.input_data.projects}
        self.hours_required_dict = {}
        self.expertise_required_dict = {}
        self.task_info = {}
        for task_obj in self.input_data.tasks:
            project = self.project_dict.get(task_obj.project_id)
            if project:
                proj_name = project.name
                task_name = task_obj.name
                task_key = (proj_name, task_name)

                if task_name not in self.tasks_per_project_name[proj_name]:
                    self.tasks_per_project_name[proj_name].append(task_name)
                    self.hours_required_dict[task_key] = task_obj.hours
                    self.expertise_required_dict[task_key] = task_obj.expertise
                    self.task_info[task_key] = task_obj # Guardar el objeto Task completo
                else:
                    print(f"ADVERTENCIA: Tarea duplicada por nombre '{task_name}' en proyecto '{proj_name}'. Se usará la primera encontrada.")
            else:
                print(f"ADVERTENCIA: Tarea '{task_obj.name}' (ID: {task_obj.id_task}) tiene un project_id ('{task_obj.project_id}') no encontrado. Será ignorada.")

        # 3. Calcular Horizonte de Planificación (days_list)
        self.start_date = self.input_data.config.start_date
        
        planning_horizon_days_config = self.input_data.config.planning_horizon_days

        if planning_horizon_days_config is not None and planning_horizon_days_config > 0:
            n_days = planning_horizon_days_config
            print(f"DEBUG BASE: Usando horizonte predefinido de {n_days} días.")
        else:
            # Calcular basado en deadlines si no se proporciona un horizonte
            latest_deadline = self.start_date
            if self.input_data.projects:
                valid_deadlines = [p.deadline for p in self.input_data.projects if isinstance(p.deadline, date)]
                if valid_deadlines:
                    latest_deadline = max(valid_deadlines)
            
            # Añadir un buffer (ej. 30 días) o un mínimo si no hay deadlines / son pasadas
            buffer_days = 30
            min_planning_days = 30 # Un mínimo razonable
            
            final_planning_date = max(latest_deadline + timedelta(days=buffer_days), self.start_date + timedelta(days=min_planning_days -1))
            
            n_days = (final_planning_date - self.start_date).days + 1
            if n_days <= 0: # Fallback por si acaso
                n_days = min_planning_days
            print(f"DEBUG BASE: Horizonte calculado: {n_days} días (hasta {final_planning_date}). Última deadline: {latest_deadline if valid_deadlines else 'N/A'}")

        self.days_list = list(range(1, n_days + 1))
        self.M_days_big_m = n_days + 10 # Big-M para restricciones de días (ej. QUBO)
        print(f"DEBUG BASE: days_list generado: 1 a {n_days}")

        # 4. Calcular Disponibilidad Numérica (availability_numeric)
        self.availability_numeric = {}
        start_weekday_idx = self.start_date.weekday() # Lunes=0, Domingo=6

        for r_obj in self.input_data.resources:
            resource_name = r_obj.name
            # r_obj.availability es un dict {"Lunes": 8, "Martes": 8, ...}
            resource_daily_availability = r_obj.availability 
            for day_num in self.days_list:
                # Calcular el día de la semana para day_num (0-6)
                current_date = self.start_date + timedelta(days=day_num - 1)
                current_weekday_idx = current_date.weekday()
                
                hours = 0 # Default a 0 horas
                if 0 <= current_weekday_idx < len(WORK_DAYS_NAMES): # Es Lunes-Viernes?
                    day_name_str = WORK_DAYS_NAMES[current_weekday_idx]
                    hours = resource_daily_availability.get(day_name_str, 0) # Obtener de Resource.availability
                
                self.availability_numeric[(resource_name, day_num)] = hours
        
        if self.resource_names and self.days_list:
            example_r = self.resource_names[0]
            example_d = self.days_list[0]
            print(f"DEBUG BASE: Ejemplo availability_numeric[({example_r},{example_d})]: {self.availability_numeric.get((example_r, example_d), 'No encontrado')}")
        else:
            print("DEBUG BASE: No hay recursos o días para mostrar ejemplo de availability_numeric.")
        print("DEBUG BASE: _prepare_common_data finalizado.")


    @abstractmethod
    def _build_model(self):
        """Define variables, objetivo y restricciones específicas del modelo concreto."""
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
        self.result = OptimizationResult(status="Not Run") # Resetear resultado
        self.solver_runtime = 0.0
        final_status = "Error" # Estado por defecto si algo falla

        try:
            print(f"--- Iniciando Modelo: {self.__class__.__name__} ---")
            
            print("Paso 1/4: Preparando datos comunes...")
            self._prepare_common_data()
            print(f"Datos comunes preparados. Horizonte: {len(self.days_list)} días. Recursos: {len(self.resource_names)}.")
            
            print("Paso 2/4: Construyendo modelo específico...")
            self._build_model()
            print("Modelo específico construido.")

            print(f"Paso 3/4: Resolviendo (Límite: {self.input_data.config.solver_time_limit}s)...")
            # _solve_model debe devolver (status_string, runtime_float)
            solve_status_str, runtime = self._solve_model()
            self.solver_runtime = runtime
            final_status = solve_status_str # Actualizar estado con el del solver
            print(f"Resolución finalizada. Estado del solver: {final_status}, Tiempo: {runtime:.2f}s.")

            print("Paso 4/4: Extrayendo resultados...")
            self._extract_results(final_status) # Pasar el estado del solver
            print("Resultados extraídos.")
            # El estado final del objeto self.result se establece dentro de _extract_results

        except ValueError as ve: # Capturar errores de validación de datos o preparación
            error_msg = f"Error de Valor/Datos: {ve}\n{traceback.format_exc()}"
            print(f"ERROR durante la optimización: {error_msg}")
            self.result = OptimizationResult(
                status="Data Error",
                solver_runtime=self.solver_runtime,
                error_message=error_msg
            )
        except Exception as e:

            error_msg = f"Error Inesperado: {e}\n{traceback.format_exc()}"
            print(f"ERROR durante la optimización: {error_msg}")
            self.result = OptimizationResult(
                status="Execution Error",
                solver_runtime=self.solver_runtime,
                error_message=error_msg
            )
        
        print(f"--- Modelo Finalizado: {self.__class__.__name__} --- Estado Final del Resultado: {self.result.status} ---")
        return self.result