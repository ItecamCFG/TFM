# ui_projects.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from data_manager import generate_id, EXPERTISE_LEVELS # Importar funciones y constantes

def display_project_management():
    """Muestra la UI revisada para gestionar proyectos y sus tareas secuenciales."""

    st.header("Gestión de Proyectos y Tareas")

    # Ensure app_data is in session state
    if 'app_data' not in st.session_state:
         st.error("Error: Datos de la aplicación no inicializados. Por favor, recarga la aplicación.")
         return

    app_data = st.session_state.app_data

    # --- Selección y Gestión de Proyectos ---
    st.subheader("Seleccionar o Crear Proyecto")

    projects_dict = {p['id']: p['name'] for p in app_data.get('projects', [])} # Use .get() for safety
    project_ids = list(projects_dict.keys())
    # project_names = list(projects_dict.values()) # Not used directly here

    col_select, col_new = st.columns([2, 3]) # Adjusted column widths for better layout

    # --- Determine the default index for the selectbox ---
    # We want to keep the currently selected project if it exists and is valid.
    # OR select a newly added project if the flag is set.
    # OR default to the first project if none of the above.
    default_index = None
    project_to_select_id = None # Variable to hold the ID we want to select

    # Priority 1: Was a new project just added?
    # Check the flag set in the *previous* run by the form submission
    if 'select_newly_added_project_id' in st.session_state and st.session_state.select_newly_added_project_id in project_ids:
        project_to_select_id = st.session_state.select_newly_added_project_id
        # Clear the flag immediately so it only triggers selection once
        del st.session_state.select_newly_added_project_id
        # The default_index will be calculated below based on project_to_select_id

    # Priority 2: Is there a project ID already stored in the selector's state that is still valid?
    # This handles cases where the user previously selected a project, or the state was initialized.
    elif 'project_selector' in st.session_state and st.session_state.project_selector in project_ids:
         project_to_select_id = st.session_state.project_selector
         # The default_index will be calculated below based on project_to_select_id

    # Priority 3: If no specific project ID is determined yet, and there are projects, default to the first one
    elif project_ids:
        project_to_select_id = project_ids[0]
        # Optionally, update the selector state here if defaulting to the first project
        # This is safe *before* the widget is called if the state didn't have a valid value
        if 'project_selector' not in st.session_state or st.session_state.project_selector != project_to_select_id:
             st.session_state.project_selector = project_to_select_id


    # Now, calculate the default_index based on the determined project_to_select_id
    if project_to_select_id is not None and project_to_select_id in project_ids:
         default_index = project_ids.index(project_to_select_id)
         # Ensure st.session_state.project_selector is set correctly for this run *before* the selectbox
         # This handles the case where we defaulted to the first project (Priority 3)
         st.session_state.project_selector = project_to_select_id
    elif not project_ids:
         # If no projects exist, the selectbox should have no options, index should be None
         default_index = None
         st.session_state.project_selector = None # Ensure state is None if no projects


    with col_select:
        # Pass the calculated default_index to the selectbox
        selected_project_id = st.selectbox(
            "Selecciona un Proyecto para ver/editar sus tareas",
            options=project_ids,
            format_func=lambda pid: projects_dict.get(pid, "Desconocido") if pid else "Selecciona un proyecto", # Handle None case in format_func
            index=default_index, # <-- Use the calculated default_index here
            key="project_selector" # Keep the key - this widget updates st.session_state.project_selector
        )
        # The value selected by the user will override the default_index selection
        # and update st.session_state.project_selector at the end of the script run.
        # We will use st.session_state.project_selector below for consistency.


    with col_new:
        with st.expander("➕ Añadir Nuevo Proyecto"):
            # Added keys to form widgets for better state management if needed later
            with st.form("new_project_form", clear_on_submit=True): # Added clear_on_submit
                new_project_name = st.text_input("Nombre del Proyecto*", key="new_proj_name_input")
                new_project_deadline = st.date_input(
                    "Fecha Límite (Deadline)*",
                    min_value=date.today(),
                    value=date.today() + timedelta(days=30), # Default a 1 mes
                    key="new_proj_deadline_input"
                )
                submitted = st.form_submit_button("Añadir Proyecto")
                if submitted:
                    if new_project_name and new_project_deadline:
                        # Check for duplicate name (case-insensitive comparison recommended)
                        existing_names = [p['name'].lower() for p in app_data.get('projects', [])]
                        if new_project_name.lower() in existing_names:
                            st.warning(f"El proyecto '{new_project_name}' ya existe.")
                        else:
                            project_id = generate_id("proj_")
                            app_data['projects'].append({
                                'id': project_id,
                                'name': new_project_name,
                                'deadline': new_project_deadline # date object is fine here
                            })
                            st.success(f"Proyecto '{new_project_name}' añadido.")

                            # --- Set the *signal* state variable *instead* of project_selector ---
                            # This variable tells the *next* run which project to select.
                            st.session_state.select_newly_added_project_id = project_id # <-- Use the signal variable

                            st.rerun() # Trigger a rerun to update the display and select the new project

                    else:
                        st.warning("Por favor, completa el nombre y la fecha límite.")

    st.divider()

    # --- Gestión de Tareas del Proyecto Seleccionado ---
    # Use the value from st.session_state.project_selector which is now reliably set *before*
    # the conditional block runs in the current script execution.
    currently_selected_project_id = st.session_state.get('project_selector')

    # Only show project/task details if a project is actually selected and found in data
    if currently_selected_project_id and currently_selected_project_id in project_ids:
        # Find the full project data for the selected project ID
        project_data = next((p for p in app_data['projects'] if p['id'] == currently_selected_project_id), None)

        if project_data:
            st.subheader(f"Tareas del Proyecto: {project_data.get('name', 'Desconocido')}") # Use .get()
            # Format deadline display
            deadline_str = project_data.get('deadline')
            if isinstance(deadline_str, date): # Check if it's already a date object
                deadline_display = deadline_str.strftime('%Y-%m-%d')
            elif isinstance(deadline_str, str): # If it's a string, try to format
                 try:
                     deadline_display = date.fromisoformat(deadline_str).strftime('%Y-%m-%d')
                 except (ValueError, TypeError):
                     deadline_display = 'Formato inválido'
            else:
                deadline_display = 'No definida'

            st.caption(f"Fecha Límite del Proyecto: {deadline_display}")


            # Mostrar / Editar Datos del Proyecto
            with st.expander("Editar Datos del Proyecto"):
                 # Ensure form key is unique based on the selected project
                 with st.form(f"edit_project_form_{currently_selected_project_id}"):
                     edited_name = st.text_input("Nombre Proyecto", value=project_data.get('name', ''), key=f"edit_proj_name_{currently_selected_project_id}") # Use .get()
                     # Handle date input value - needs to be a date object or None
                     current_deadline_value = project_data.get('deadline')
                     if isinstance(current_deadline_value, str):
                         try:
                            current_deadline_value = date.fromisoformat(current_deadline_value)
                         except (ValueError, TypeError):
                            current_deadline_value = None # Or date.today() as a fallback? Let's use None for now

                     edited_deadline = st.date_input(
                         "Fecha Límite",
                         value=current_deadline_value,
                         min_value=date.today(), # Allow changing to today or later
                         key=f"edit_proj_deadline_{currently_selected_project_id}"
                     )

                     col_save, col_del = st.columns(2)
                     save_proj = col_save.form_submit_button("Guardar Cambios Proyecto")
                     delete_proj = col_del.form_submit_button("🗑️ Eliminar Proyecto Completo")

                     if save_proj:
                         if edited_name:
                              # Check for duplicate name excluding the current project
                              other_names = [p['name'].lower() for p in app_data.get('projects', []) if p['id'] != currently_selected_project_id] # Use .get()
                              if edited_name.lower() in other_names:
                                  st.warning(f"Ya existe otro proyecto llamado '{edited_name}'.")
                              else:
                                  # Find the project in the original app_data list by ID and update it
                                  for p in app_data.get('projects', []): # Use .get()
                                      if p['id'] == currently_selected_project_id:
                                          p['name'] = edited_name
                                          # Ensure deadline is stored as date object if possible
                                          p['deadline'] = edited_deadline if isinstance(edited_deadline, date) else None
                                          break # Found and updated
                                  st.success("Datos del proyecto actualizados.")
                                  st.rerun() # Rerun to refresh display

                         else:
                              st.warning("El nombre del proyecto no puede estar vacío.")

                     if delete_proj:
                          # Confirm deletion? Add a confirmation dialog if needed.
                          # Eliminar proyecto de la lista
                          app_data['projects'] = [p for p in app_data.get('projects', []) if p['id'] != currently_selected_project_id] # Use .get()
                          # Eliminar tareas asociadas a este proyecto
                          app_data['tasks'] = [t for t in app_data.get('tasks', []) if t.get('project_id') != currently_selected_project_id] # Use .get()
                          st.success(f"Proyecto '{project_data.get('name', 'Seleccionado')}' y sus tareas eliminadas.")
                          st.session_state.project_selector = None # Deselect the project
                          # Clear the new project selection flag just in case (shouldn't be set here, but for safety)
                          if 'select_newly_added_project_id' in st.session_state:
                              del st.session_state.select_newly_added_project_id
                          st.rerun() # Rerun to update the display (selectbox will show empty or first project)


            # Obtener y ordenar tareas del proyecto seleccionado
            # Filter tasks belonging to the currently selected project ID
            project_tasks = sorted(
                [task for task in app_data.get('tasks', []) if task.get('project_id') == currently_selected_project_id], # Use .get()
                key=lambda t: t.get('sequence', 0) # Order by sequence, default to 0 if sequence is missing
            )

            # --- Mostrar y gestionar Tareas ---
            st.markdown("**Tareas (en orden de ejecución):**")

            if not project_tasks:
                st.info("Este proyecto aún no tiene tareas. Añádelas a continuación.")
            else:
                 # Using columns to display tasks
                 # Headers for the task list
                 cols_header = st.columns([0.1, 0.4, 0.15, 0.15, 0.07, 0.07, 0.06])
                 cols_header[0].write("**#**")
                 cols_header[1].write("**Tarea**")
                 cols_header[2].write("**Horas**")
                 cols_header[3].write("**Expertis**")
                 cols_header[4].write("") # Up arrow column
                 cols_header[5].write("") # Down arrow column
                 cols_header[6].write("") # Delete column


                 for i, task in enumerate(project_tasks):
                     # Create unique columns for each task item
                     cols = st.columns([0.1, 0.4, 0.15, 0.15, 0.07, 0.07, 0.06]) # Adjust widths as needed
                     task_id = task.get('id', f"temp_task_{i}") # Ensure task has an ID, fallback to temp key

                     cols[0].write(f"**{task.get('sequence', i+1)}.**") # Display sequence or index + 1
                     cols[1].write(task.get('name', 'Sin Nombre')) # Use .get() for safety
                     cols[2].write(f"{task.get('hours', 0)} h") # Use .get()
                     cols[3].write(f"{task.get('expertise', 'N/A')}") # Use .get()

                     # Botones de acción - Need unique keys for each button
                     # Keys must be unique across all widgets in the app, including those within iterations.
                     # Incorporate task_id and currently_selected_project_id into the key.
                     if cols[4].button("⬆️", key=f"up_task_{task_id}_proj_{currently_selected_project_id}", help="Mover tarea arriba") and i > 0:
                          # Find the task and previous task in the original app_data list and swap their sequences
                          # This is more robust than swapping in the temporary project_tasks list
                          task_to_move = next((t for t in app_data.get('tasks', []) if t.get('id') == task_id), None) # Use .get()
                          task_before = next((t for t in app_data.get('tasks', []) if t.get('id') == project_tasks[i-1].get('id')), None) # Find the actual object in app_data, use .get()

                          if task_to_move and task_before:
                              # Swap sequence numbers
                              seq1 = task_to_move.get('sequence', i + 1)
                              seq2 = task_before.get('sequence', i)
                              task_to_move['sequence'] = seq2
                              task_before['sequence'] = seq1
                              st.rerun() # Rerun to show the new order

                     if cols[5].button("⬇️", key=f"down_task_{task_id}_proj_{currently_selected_project_id}", help="Mover tarea abajo") and i < len(project_tasks) - 1:
                         # Find the task and next task in the original app_data list and swap their sequences
                         task_to_move = next((t for t in app_data.get('tasks', []) if t.get('id') == task_id), None) # Use .get()
                         task_after = next((t for t in app_data.get('tasks', []) if t.get('id') == project_tasks[i+1].get('id')), None) # Find the actual object in app_data, use .get()

                         if task_to_move and task_after:
                             # Swap sequence numbers
                             seq1 = task_to_move.get('sequence', i + 1)
                             seq2 = task_after.get('sequence', i + 2)
                             task_to_move['sequence'] = seq2
                             task_after['sequence'] = seq1
                             st.rerun() # Rerun to show the new order


                     if cols[6].button("🗑️", key=f"del_task_{task_id}_proj_{currently_selected_project_id}", help="Eliminar tarea"):
                          # Filter out the task to be deleted from the app_data list
                          app_data['tasks'] = [t for t in app_data.get('tasks', []) if t.get('id') != task_id] # Use .get()

                          # Re-sequence the remaining tasks for this specific project
                          remaining_project_tasks = sorted(
                              [t for t in app_data.get('tasks', []) if t.get('project_id') == currently_selected_project_id], # Use .get() and currently_selected_project_id
                              key=lambda t: t.get('sequence', 0)
                          )

                          # Rebuild the entire tasks list with re-sequenced tasks for the current project
                          rebuilt_tasks = []
                          current_seq_num = 1
                          # Add tasks from *other* projects first
                          for t in app_data['tasks']: # app_data['tasks'] is already filtered for deleted task
                             if t.get('project_id') != currently_selected_project_id:
                                 rebuilt_tasks.append(t)

                          # Then add tasks from the *current* project, with updated sequences
                          for t in remaining_project_tasks: # Use the sorted list of remaining tasks for this project
                              # Create a copy or update the original object if necessary, but re-sequencing means rebuilding is cleaner
                              t['sequence'] = current_seq_num
                              rebuilt_tasks.append(t)
                              current_seq_num += 1

                          app_data['tasks'] = rebuilt_tasks # Replace the old list

                          st.success(f"Tarea '{task.get('name', 'Seleccionada')}' eliminada.") # Use .get()
                          st.rerun() # Rerun to update the task list display


            # --- Formulario para Añadir Nueva Tarea ---
            st.markdown("**Añadir Nueva Tarea al Final de la Secuencia:**")
            # Ensure form key is unique per selected project
            with st.form(f"new_task_form_{currently_selected_project_id}", clear_on_submit=True):
                # Ensure widget keys are unique per selected project
                task_name = st.text_input("Nombre de la Tarea*", key=f"new_task_name_{currently_selected_project_id}")
                task_hours = st.number_input("Horas Requeridas*", min_value=1, step=1, key=f"new_task_hours_{currently_selected_project_id}")
                task_expertise = st.selectbox("Expertis Mínimo Requerido*", EXPERTISE_LEVELS, index=0, key=f"new_task_exp_{currently_selected_project_id}")

                submit_task = st.form_submit_button("Añadir Tarea")
                if submit_task:
                    if task_name and task_hours > 0 and task_expertise:
                         # Check for duplicate task name within *this project*
                         if any(t.get('name') == task_name and t.get('project_id') == currently_selected_project_id for t in app_data.get('tasks', [])): # Use .get()
                            st.warning(f"Ya existe una tarea llamada '{task_name}' en este proyecto.")
                         else:
                            new_task_id = generate_id("task_")
                            # Calculate the next sequence for tasks *only within the current project*
                            project_tasks_for_sequence = [t for t in app_data.get('tasks', []) if t.get('project_id') == currently_selected_project_id] # Use .get()
                            # Find the maximum sequence number among existing tasks for this project
                            max_sequence = max((t.get('sequence', 0) for t in project_tasks_for_sequence), default=0) # Use default=0 for empty list
                            next_sequence = max_sequence + 1

                            app_data['tasks'].append({
                                'id': new_task_id,
                                'project_id': currently_selected_project_id, # Associate task with the selected project
                                'name': task_name,
                                'hours': task_hours,
                                'expertise': task_expertise,
                                'sequence': next_sequence # Assign the calculated sequence
                            })
                            st.success(f"Tarea '{task_name}' añadida al proyecto.")
                            st.rerun() # Rerun to update the task list display

                    else:
                        st.warning("Completa todos los campos de la tarea.")
        else:
             # This case should ideally not happen if currently_selected_project_id is in project_ids
             st.error("Error interno: No se pudieron cargar los datos del proyecto seleccionado.")

    # Optional: Handle the case where no project is selected or projects list is empty
    elif project_ids:
        # This happens initially if index=None and no state was set, or after deleting the selected project
        st.info("Selecciona un proyecto para ver o añadir tareas.")
    else:
        # This happens when there are absolutely no projects in the data
        st.info("No hay proyectos creados aún. Usa el formulario de 'Añadir Nuevo Proyecto' para empezar.")