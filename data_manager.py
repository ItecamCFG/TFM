# data_manager.py
import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import date, datetime

DATA_FILE = "app_data_v2.json" # Nuevo nombre para evitar conflictos
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
EXPERTISE_COSTS = {"Junior": 30, "Senior": 45, "Experto": 60}
EXPERTISE_LEVELS = list(EXPERTISE_COSTS.keys())

def generate_id(prefix=""):
    """Genera un ID único."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"

def get_default_availability():
    """Devuelve la disponibilidad por defecto (8h L-V)."""
    return {day: 8 for day in DAYS}

def get_initial_state():
    """Define la estructura inicial del estado v2."""
    return {
        'projects': [], # Lista de diccionarios de proyectos [{'id': '...', 'name': '...', 'deadline': 'YYYY-MM-DD'}]
        'tasks': [], # Lista de diccs de tareas [{'id_task': '...', 'project_id': '...', 'name': '...', 'hours': ..., 'expertise': '...', 'sequence': ...}]
        'resources': [] # Lista de diccs de recursos [{'name': '...', 'expertise': '...', 'cost': ..., 'Lunes': 8, ...}]
    }

def import_excel_to_app_data(excel_file: str) -> dict:
    """Importa un archivo Excel con hojas 'Proyectos', 'Tareas' y 'Recursos' al formato app_data."""
    try:
        xls = pd.ExcelFile(excel_file)
        df_proj = pd.read_excel(xls, 'Proyectos')
        df_tasks = pd.read_excel(xls, 'Tareas')
        df_resources = pd.read_excel(xls, 'Recursos')

        # --- Procesar Proyectos ---
        projects = []
        for _, row in df_proj.iterrows():
            projects.append({
                "id": str(row["ID Proyecto"]).strip(),
                "name": str(row["Nombre Proyecto"]).strip(),
                "deadline": datetime.strptime(str(row["Fecha Límite"]), "%Y-%m-%d").date() if not pd.isna(row["Fecha Límite"]) else None
            })

        # --- Procesar Tareas ---
        tasks = []
        for _, row in df_tasks.iterrows():
            tasks.append({
                "id_task": str(row["ID Tarea"]).strip(),
                "project_id": str(row["ID Proyecto"]).strip(),
                "name": str(row["Nombre Tarea"]).strip(),
                "hours": float(row["Horas"]),
                "expertise": str(row["Expertise"]).strip(),
                "sequence": int(row["Secuencia"]) if not pd.isna(row["Secuencia"]) else 0
            })

        # --- Procesar Recursos ---
        resources = []
        for _, row in df_resources.iterrows():
            expertise = str(row["Expertise"]).strip()
            resource = {
                "name": str(row["Nombre"]).strip(),
                "expertise": expertise,
                "cost": EXPERTISE_COSTS.get(expertise, 0)
            }
            # Añadir disponibilidad para Lunes a Viernes
            for day in DAYS:
                resource[day] = float(row[day]) if day in row and not pd.isna(row[day]) else 0.0
            resources.append(resource)

        return {
            "projects": projects,
            "tasks": tasks,
            "resources": resources
        }

    except Exception as e:
        st.error(f"Error al importar Excel: {e}")
        return {
            "projects": [],
            "tasks": [],
            "resources": []
        }


def load_data():
    """Carga los datos v2 desde el archivo JSON."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validar y completar estructura si es necesario (simplificado)
                data.setdefault('projects', [])
                data.setdefault('tasks', [])
                data.setdefault('resources', [])
                # Asegurar costes correctos y disponibilidad completa en recursos
                for res in data.get('resources', []):
                    res['cost'] = EXPERTISE_COSTS.get(res.get('expertise'), 0)
                    for day in DAYS:
                        res.setdefault(day, get_default_availability().get(day, 0)) # Asegurar todos los días
                st.success(f"Datos cargados desde {DATA_FILE}")
                return data
        except Exception as e:
            st.error(f"Error al cargar {DATA_FILE}: {e}. Usando estado inicial.")
            return get_initial_state()
    else:
        st.info("No se encontró archivo de datos. Empezando con estado vacío.")
        return get_initial_state()

def save_data(data):
    """Guarda el estado actual v2 en el archivo JSON."""
    try:
        # Asegurarse de que los costes son correctos antes de guardar
        for res in data.get('resources', []):
             res['cost'] = EXPERTISE_COSTS.get(res.get('expertise'), 0)

        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            # Convertir fechas deadline a string para JSON
            data_to_save = data.copy()
            data_to_save['projects'] = [
                {**p, 'deadline': p['deadline'].isoformat() if isinstance(p.get('deadline'), date) else p.get('deadline')}
                for p in data.get('projects', [])
            ]
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        st.success(f"Datos guardados en {DATA_FILE}")
    except Exception as e:
        st.error(f"Error al guardar datos: {e}")

def initialize_session_state():
    """Inicializa st.session_state con datos v2."""
    if 'app_data' not in st.session_state:
        loaded_data = load_data()
        # Convertir deadlines de string a date al cargar
        for project in loaded_data.get('projects', []):
            if isinstance(project.get('deadline'), str):
                try:
                    project['deadline'] = date.fromisoformat(project['deadline'])
                except (ValueError, TypeError):
                    project['deadline'] = None # O manejar el error de otra forma
        st.session_state['app_data'] = loaded_data

    # Inicializar claves si faltan después de cargar (por si el archivo estaba incompleto)
    st.session_state.app_data.setdefault('projects', [])
    st.session_state.app_data.setdefault('tasks', [])
    st.session_state.app_data.setdefault('resources', [])