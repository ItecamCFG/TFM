# optimization/genetic_model.py
import random
from datetime import date, timedelta
from optimization.base_model import OptimizationModel, WORK_DAYS_NAMES
from optimization.data_models import OptimizationResult, Task, Resource, OptimizationConfig
import copy
from joblib import Parallel, delayed
import math
import time
from typing import List, Dict, Tuple, Any, Optional

class GeneticMakespanPlanner(OptimizationModel):
    """Planificador basado en Algoritmo Genético para minimizar el Makespan."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.POP_SIZE = 50
        self.N_GEN = 100
        self.MUT_PROB = 0.15
        self.CROSS_PROB = 0.8
        self.ELITE_SIZE = 5
        # Ejecutar en un solo proceso por defecto para evitar problemas con
        # el ResourceTracker de joblib en algunos entornos Windows.
        self.N_JOBS = 1

        if hasattr(self.input_data.config, 'ga_params') and isinstance(self.input_data.config.ga_params, dict):
            ga_params = self.input_data.config.ga_params
            self.POP_SIZE = ga_params.get('pop_size', self.POP_SIZE)
            self.N_GEN = ga_params.get('n_gen', self.N_GEN)
            self.MUT_PROB = ga_params.get('mut_prob', self.MUT_PROB)
            self.CROSS_PROB = ga_params.get('cross_prob', self.CROSS_PROB)
            self.ELITE_SIZE = ga_params.get('elite_size', self.ELITE_SIZE)
            self.N_JOBS = ga_params.get('n_jobs', self.N_JOBS)

        self.task_objects_by_id: Dict[str, Task] = {}

    def _build_model(self):
        """Preparación previa para el GA."""
        # Asegurarse de que los datos comunes ya se prepararon en _prepare_common_data
        # (llamado desde solve()). Aquí calculamos algunos mapeos auxiliares.
        self.resource_expertise_map = {r.name: r.expertise for r in self.input_data.resources}
        # M_days es útil como horizonte máximo para el algoritmo
        self.M_days = len(self.days_list)
        self.project_id_to_name = {p.id: p.name for p in self.input_data.projects}

    def _solve_model(self, time_limit: Optional[int] = None) -> tuple[str, float]:
        """Ejecuta el GA para minimizar el makespan."""
        # Los datos comunes y los mapeos auxiliares ya están preparados antes de
        # esta llamada a través de solve().

        all_tasks_flat: List[Task] = []
        for project_name in self.project_names:
            for task_name in self.tasks_per_project_name[project_name]:
                all_tasks_flat.append(self.task_dict[task_name])
        self.task_objects_by_id = {task.id_task: task for task in all_tasks_flat}

        if not hasattr(self, 'predecessors') or not self.predecessors:
            self._calculate_predecessors()

        population = [self._generate_individual_permutation(all_tasks_flat) for _ in range(self.POP_SIZE)]

        best_solution_makespan = float('inf')
        best_solution_assignment = {}
        best_solution_task_completion_days = {}
        best_individual = None

        final_status_str = "Feasible (GA)"
        start_solve_time = time.time()

        for gen in range(self.N_GEN):
            if time_limit is not None and (time.time() - start_solve_time) > time_limit:
                print(f"DEBUG: Límite de tiempo ({time_limit}s) alcanzado en la generación {gen}.")
                final_status_str = "Timelimit (GA)"
                break

            scored_individuals = Parallel(n_jobs=self.N_JOBS)(
                delayed(self._evaluate_and_decode)(ind) for ind in population
            )
            scored_individuals.sort(key=lambda x: x[1])

            current_best_makespan = scored_individuals[0][1]
            if current_best_makespan < best_solution_makespan:
                best_solution_makespan = current_best_makespan
                best_solution_assignment = scored_individuals[0][2]
                best_solution_task_completion_days = scored_individuals[0][3]
                best_individual = scored_individuals[0][0]
                print(f"DEBUG: Gen {gen+1}: Nueva mejor solución. Makespan: {best_solution_makespan:.2f}")

            next_gen = []
            next_gen.extend([ind[0] for ind in scored_individuals[:self.ELITE_SIZE]])

            while len(next_gen) < self.POP_SIZE:
                parent1 = self._select_one(scored_individuals)
                parent2 = self._select_one(scored_individuals)

                if random.random() < self.CROSS_PROB:
                    child1, child2 = self._crossover_ox(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                if random.random() < self.MUT_PROB:
                    child1 = self._mutate_swap(child1)
                if random.random() < self.MUT_PROB:
                    child2 = self._mutate_swap(child2)

                next_gen.append(child1)
                if len(next_gen) < self.POP_SIZE:
                    next_gen.append(child2)

            population = next_gen[:self.POP_SIZE]

        if best_individual:
            _, final_makespan, final_assignment, final_task_completion_days = self._evaluate_and_decode(best_individual)
            if final_makespan <= best_solution_makespan:
                best_solution_makespan = final_makespan
                best_solution_assignment = final_assignment
                best_solution_task_completion_days = final_task_completion_days

        self.result = OptimizationResult(
            status=final_status_str,
            solver_runtime=time.time() - start_solve_time,
            objective_value=best_solution_makespan,
            makespan=best_solution_makespan,
            assignment=best_solution_assignment,
            task_completion_days=best_solution_task_completion_days,
        )
        return final_status_str, self.result.solver_runtime

    def _extract_results(self, status: str):
        if self.result:
            self.result.status = status

    def _calculate_predecessors(self):
        self.predecessors: Dict[str, List[str]] = {task.id_task: [] for task in self.input_data.tasks}
        task_sequence_map = {}
        for task in self.input_data.tasks:
            task_sequence_map[(task.project_id, task.sequence)] = task.id_task
        for task in self.input_data.tasks:
            if task.sequence > 1:
                predecessor_sequence = task.sequence - 1
                predecessor_id = task_sequence_map.get((task.project_id, predecessor_sequence))
                if predecessor_id:
                    self.predecessors[task.id_task].append(predecessor_id)

    def _generate_individual_permutation(self, all_tasks: List[Task]) -> List[str]:
        task_ids = [task.id_task for task in all_tasks]
        random.shuffle(task_ids)
        return task_ids

    def _evaluate_and_decode(self, individual_permutation: List[str]) -> Tuple[List[str], float, Dict, Dict]:
        task_start_times: Dict[str, float] = {}
        task_end_times: Dict[str, float] = {}
        resource_availability: Dict[Tuple[str, int], float] = copy.deepcopy(self.availability_numeric)
        assignment_result: Dict[Tuple[str, str, str, int], float] = {}

        for r_name in self.resource_names:
            for day_num in self.days_list:
                if (r_name, day_num) not in resource_availability:
                    resource_availability[(r_name, day_num)] = 0.0

        for task_id in individual_permutation:
            task: Task = self.task_objects_by_id[task_id]
            # Convertir project_id a nombre para mantener consistencia con otros modelos
            project_name = self.project_id_to_name.get(task.project_id, task.project_id)

            earliest_start_time = 0.0
            for pred_id in self.predecessors.get(task_id, []):
                if pred_id not in task_end_times or task_end_times[pred_id] == float('inf'):
                    earliest_start_time = float('inf')
                    break
                earliest_start_time = max(earliest_start_time, task_end_times[pred_id])

            if earliest_start_time == float('inf'):
                task_start_times[task_id] = float('inf')
                task_end_times[task_id] = float('inf')
                continue

            task_remaining_hours = task.hours
            current_day_num = max(1, math.ceil(earliest_start_time))
            task_start_day_numeric: Optional[int] = None

            while task_remaining_hours > 0:
                if current_day_num > self.M_days:
                    task_end_times[task_id] = float('inf')
                    task_start_times[task_id] = float('inf')
                    break

                hours_assigned_on_this_day = 0.0
                eligible_resources = [r_name for r_name in self.resource_names if self.resource_expertise_map.get(r_name) == task.expertise]
                random.shuffle(eligible_resources)

                for r_name in eligible_resources:
                    if task_remaining_hours <= 0:
                        break

                    available_for_resource_day = resource_availability.get((r_name, current_day_num), 0.0)

                    if available_for_resource_day > 0:
                        if task_start_day_numeric is None:
                            task_start_day_numeric = current_day_num

                        hours_to_assign = min(task_remaining_hours, available_for_resource_day)

                        assignment_result[(project_name, task_id, r_name, current_day_num)] = \
                            assignment_result.get((project_name, task_id, r_name, current_day_num), 0.0) + hours_to_assign

                        resource_availability[(r_name, current_day_num)] -= hours_to_assign
                        task_remaining_hours -= hours_to_assign
                        hours_assigned_on_this_day += hours_to_assign

                if hours_assigned_on_this_day == 0 and task_remaining_hours > 0:
                    current_day_num += 1
                elif task_remaining_hours > 0:
                    current_day_num += 1

            if task_end_times.get(task_id) != float('inf'):
                if task_start_day_numeric is not None:
                    task_start_times[task_id] = task_start_day_numeric
                    task_end_times[task_id] = current_day_num - 1
                else:
                    task_start_times[task_id] = earliest_start_time
                    task_end_times[task_id] = earliest_start_time

        makespan = 0.0
        valid_task_end_times = [end_time for end_time in task_end_times.values() if end_time != float('inf')]
        if valid_task_end_times:
            makespan = max(valid_task_end_times)
        else:
            makespan = float('inf')

        task_completion_days_result = {}
        for tid, end_time in task_end_times.items():
            proj_name = self.project_id_to_name.get(self.task_dict[tid].project_id,
                                                    self.task_dict[tid].project_id)
            if end_time != float('inf'):
                task_completion_days_result[(proj_name, tid)] = int(math.ceil(end_time))
            else:
                task_completion_days_result[(proj_name, tid)] = None

        if makespan == float('inf'):
            makespan = self.M_days + 1000

        return individual_permutation, makespan, assignment_result, task_completion_days_result

    def _select_one(self, scored_population: List[Tuple[List[str], float, Dict, Dict]]) -> List[str]:
        tournament_size = max(2, int(0.1 * len(scored_population)))
        contestants = random.sample(scored_population, tournament_size)
        contestants.sort(key=lambda x: x[1])
        return contestants[0][0]

    def _crossover_ox(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child1 = [None] * size
        child2 = [None] * size
        child1[a:b] = parent1[a:b]
        child2[a:b] = parent2[a:b]
        fill_pos1 = [i for i in range(size) if not child1[i]]
        fill_pos2 = [i for i in range(size) if not child2[i]]
        fill_values1 = [item for item in parent2 if item not in child1]
        fill_values2 = [item for item in parent1 if item not in child2]
        for pos, val in zip(fill_pos1, fill_values1):
            child1[pos] = val
        for pos, val in zip(fill_pos2, fill_values2):
            child2[pos] = val
        return child1, child2

    def _mutate_swap(self, individual: List[str]) -> List[str]:
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual
