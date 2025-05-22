# data_models.py
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Optional, Any, Tuple

# --- Clases para Datos de Entrada Fundamentales ---

@dataclass
class Resource:
    name: str
    expertise: str # Considerar Enum: ("Junior", "Senior", "Experto")
    cost: float = 0.0 # Se puede recalcular al crear o antes de pasar al modelo
    availability: Dict[str, int] = field(default_factory=dict) # {"Lunes": 8, ...}

@dataclass
class Task:
    id_task: str
    project_id: str
    name: str
    hours: float
    expertise: str # Considerar Enum
    sequence: int


@dataclass
class Project:
    id: str
    name: str
    deadline: Optional[date] = None
    # Podríamos añadir una lista de Task IDs aquí, pero mantenerlas separadas
    # en la entrada del modelo es más similar a lo actual

# --- Clases para Entrada/Salida de Optimización ---

@dataclass
class OptimizationConfig:
    """Configuración para la ejecución del modelo."""
    start_date: date = field(default_factory=date.today)
    planning_horizon_days: Optional[int] = None  # Si se precalcula el num días
    solver_time_limit: int = 300  # Segundos
    ga_params: Optional[Dict[str, Any]] = None  # Parámetros para el algoritmo genético
    # Añadir otros parámetros si son necesarios (ej: factor penalización si se usa)

@dataclass
class OptimizationInput:
    """Datos de entrada estructurados para los modelos."""
    projects: List[Project]
    tasks: List[Task]
    resources: List[Resource]
    config: OptimizationConfig

@dataclass
class OptimizationResult:
    """Resultados estructurados de la optimización."""
    status: str # "Optimal", "Timelimit", "Infeasible", "Error", etc.
    solver_runtime: float = 0.0
    objective_value: Optional[float] = None # Valor del objetivo (coste o makespan)
    makespan: Optional[float] = None # Día de finalización
    total_cost: Optional[float] = None # Coste total calculado
    assignment: Dict[Tuple[str, str, str, int], float] = field(default_factory=dict) # {(p_name, t_name, r_name, d_num): hours}
    task_completion_days: Dict[Tuple[str, str], float] = field(default_factory=dict) # {(p_name, t_name): day_num}
    error_message: Optional[str] = None
    log: Optional[str] = None # Para guardar logs del solver si se capturan