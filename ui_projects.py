# ui_projects.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from data_manager import generate_id, EXPERTISE_LEVELS # Importar funciones y constantes

def display_project_management():
    """Muestra la UI revisada para gestionar proyectos y sus tareas secuenciales."""

    st.header("Gestión de Proyectos y Tareas")

    app_data = st.session_state.app_data

    # --- Selección y Gestión de Proyectos ---
    st.subheader("Seleccionar o Crear Proyecto")

    projects_dict = {p['id']: p['name'] for p in app_data['projects']}
    project_ids = list(projects_dict.keys())
    project_names = list(projects_dict.values())

    col_select, col_new = st.columns(2)

    with col_select:
        selected_project_id = st.selectbox(
            "Selecciona un Proyecto para ver/editar sus tareas",
            options=project_ids,
            format_func=lambda pid: projects_dict.get(pid, "Desconocido"),
            index=None, # No seleccionar ninguno por defecto
            key="project_selector"
        )

    with col_new:
        with st.expander("➕ Añadir Nuevo Proyecto"):
            with st.form("new_project_form"):
                new_project_name = st.text_input("Nombre del Proyecto*")
                new_project_deadline = st.date_input(
                    "Fecha Límite (Deadline)*",
                    min_value=date.today(),
                    value=date.today() + timedelta(days=30) # Default a 1 mes
                )
                submitted = st.form_submit_button("Añadir Proyecto")
                if submitted:
                    if new_project_name and new_project_deadline:
                        if new_project_name not in project_names:
                            project_id = generate_id("proj_")
                            app_data['projects'].append({
                                'id': project_id,
                                'name': new_project_name,
                                'deadline': new_project_deadline
                            })
                            st.success(f"Proyecto '{new_project_name}' añadido.")
                            # Limpiar selector para evitar confusión o seleccionar el nuevo
                            st.session_state.project_selector = project_id
                            st.rerun()
                        else:
                            st.warning(f"El proyecto '{new_project_name}' ya existe.")
                    else:
                        st.warning("Por favor, completa el nombre y la fecha límite.")

    st.divider()

    # --- Gestión de Tareas del Proyecto Seleccionado ---
    if selected_project_id:
        project_data = next((p for p in app_data['projects'] if p['id'] == selected_project_id), None)
        if project_data:
            st.subheader(f"Tareas del Proyecto: {project_data['name']}")
            st.caption(f"Fecha Límite del Proyecto: {project_data['deadline'].strftime('%Y-%m-%d') if project_data.get('deadline') else 'No definida'}")

            # Mostrar / Editar Datos del Proyecto
            with st.expander("Editar Datos del Proyecto"):
                 with st.form(f"edit_project_{selected_project_id}"):
                    edited_name = st.text_input("Nombre Proyecto", value=project_data['name'])
                    edited_deadline = st.date_input("Fecha Límite", value=project_data.get('deadline'), min_value=date.today())
                    col_save, col_del = st.columns(2)
                    save_proj = col_save.form_submit_button("Guardar Cambios Proyecto")
                    delete_proj = col_del.form_submit_button("🗑️ Eliminar Proyecto Completo")

                    if save_proj:
                         if edited_name:
                             # Verificar nombre duplicado (excluyendo el actual)
                             other_names = [p['name'] for p in app_data['projects'] if p['id'] != selected_project_id]
                             if edited_name in other_names:
                                 st.warning(f"Ya existe otro proyecto llamado '{edited_name}'.")
                             else:
                                 project_data['name'] = edited_name
                                 project_data['deadline'] = edited_deadline
                                 st.success("Datos del proyecto actualizados.")
                                 st.rerun()
                         else:
                             st.warning("El nombre del proyecto no puede estar vacío.")

                    if delete_proj:
                        # Confirmación sería ideal aquí
                        # Eliminar proyecto
                        app_data['projects'] = [p for p in app_data['projects'] if p['id'] != selected_project_id]
                        # Eliminar tareas asociadas
                        app_data['tasks'] = [t for t in app_data['tasks'] if t.get('project_id') != selected_project_id]
                        st.success(f"Proyecto '{project_data['name']}' y sus tareas eliminadas.")
                        st.session_state.project_selector = None # Deseleccionar
                        st.rerun()


            # Obtener y ordenar tareas del proyecto seleccionado
            project_tasks = sorted(
                [task for task in app_data['tasks'] if task.get('project_id') == selected_project_id],
                key=lambda t: t.get('sequence', 0) # Ordenar por secuencia
            )

            # --- Mostrar y gestionar Tareas ---
            st.markdown("**Tareas (en orden de ejecución):**")

            if not project_tasks:
                st.info("Este proyecto aún no tiene tareas. Añádelas a continuación.")
            else:
                 # Mostrar tareas con controles (Iterar o usar DataFrame/DataEditor limitado)
                 # Usaremos iteración para controles más finos (mover arriba/abajo)
                 for i, task in enumerate(project_tasks):
                    cols = st.columns([0.1, 0.4, 0.15, 0.15, 0.07, 0.07, 0.06]) # Ajustar anchos
                    cols[0].write(f"**{task.get('sequence', i+1)}.**")
                    cols[1].write(task.get('name', 'Sin Nombre'))
                    cols[2].write(f"{task.get('hours', 0)} h")
                    cols[3].write(f"{task.get('expertise', 'N/A')}")

                    # Botones de acción
                    if cols[4].button("⬆️", key=f"up_{task['id']}", help="Mover tarea arriba") and i > 0:
                         # Intercambiar secuencia con la tarea anterior
                         prev_task = project_tasks[i-1]
                         current_seq = task.get('sequence', i+1)
                         prev_seq = prev_task.get('sequence', i)
                         task['sequence'] = prev_seq
                         prev_task['sequence'] = current_seq
                         st.rerun()

                    if cols[5].button("⬇️", key=f"down_{task['id']}", help="Mover tarea abajo") and i < len(project_tasks) - 1:
                        # Intercambiar secuencia con la tarea siguiente
                        next_task = project_tasks[i+1]
                        current_seq = task.get('sequence', i+1)
                        next_seq = next_task.get('sequence', i+2)
                        task['sequence'] = next_seq
                        next_task['sequence'] = current_seq
                        st.rerun()

                    if cols[6].button("🗑️", key=f"del_{task['id']}", help="Eliminar tarea"):
                        app_data['tasks'] = [t for t in app_data['tasks'] if t.get('id') != task['id']]
                        # Opcional: Reajustar secuencias de tareas restantes
                        remaining_tasks = sorted(
                            [t for t in app_data['tasks'] if t.get('project_id') == selected_project_id],
                             key=lambda t: t.get('sequence', 0)
                        )
                        for idx, rt in enumerate(remaining_tasks):
                            rt['sequence'] = idx + 1
                        st.success(f"Tarea '{task.get('name')}' eliminada.")
                        st.rerun()

                    # Podrías añadir un botón de "Editar" que llene el formulario de abajo


            # --- Formulario para Añadir Nueva Tarea ---
            st.markdown("**Añadir Nueva Tarea al Final de la Secuencia:**")
            with st.form(f"new_task_form_{selected_project_id}", clear_on_submit=True):
                task_name = st.text_input("Nombre de la Tarea*", key=f"new_task_name_{selected_project_id}")
                task_hours = st.number_input("Horas Requeridas*", min_value=1, step=1, key=f"new_task_hours_{selected_project_id}")
                task_expertise = st.selectbox("Expertis Mínimo Requerido*", EXPERTISE_LEVELS, index=0, key=f"new_task_exp_{selected_project_id}")

                submit_task = st.form_submit_button("Añadir Tarea")
                if submit_task:
                    if task_name and task_hours > 0 and task_expertise:
                        # Verificar si el nombre ya existe en este proyecto
                        if any(t['name'] == task_name for t in project_tasks):
                            st.warning(f"Ya existe una tarea llamada '{task_name}' en este proyecto.")
                        else:
                            new_task_id = generate_id("task_")
                            # Calcular la siguiente secuencia
                            next_sequence = (project_tasks[-1].get('sequence', 0) + 1) if project_tasks else 1
                            app_data['tasks'].append({
                                'id': new_task_id,
                                'project_id': selected_project_id,
                                'name': task_name,
                                'hours': task_hours,
                                'expertise': task_expertise,
                                'sequence': next_sequence
                            })
                            st.success(f"Tarea '{task_name}' añadida al proyecto.")
                            st.rerun()
                    else:
                        st.warning("Completa todos los campos de la tarea.")