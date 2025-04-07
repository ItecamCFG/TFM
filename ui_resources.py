# ui_resources.py
import streamlit as st
import pandas as pd
from data_manager import DAYS, EXPERTISE_LEVELS, EXPERTISE_COSTS, get_default_availability # Importar funciones/constantes necesarias

def display_resource_management():
    """Muestra la UI para gestionar recursos (coste derivado) y disponibilidad."""

    st.header("Gestión de Recursos y Disponibilidad")

    app_data = st.session_state.app_data

    # --- NUEVO: Formulario Simplificado para Añadir Recursos ---
    st.subheader("Añadir Nuevo Recurso")
    with st.form("new_resource_form", clear_on_submit=True):
        new_resource_name = st.text_input("Nombre del Nuevo Recurso*")
        new_resource_expertise = st.selectbox(
            "Expertis del Nuevo Recurso*",
            options=EXPERTISE_LEVELS,
            index=None, # No seleccionar ninguno por defecto
            placeholder="Selecciona el nivel de expertis..."
        )

        submitted = st.form_submit_button("Añadir Recurso")
        if submitted:
            # Validaciones
            if not new_resource_name:
                st.warning("Por favor, introduce el nombre del recurso.")
            elif not new_resource_expertise:
                st.warning("Por favor, selecciona el expertis del recurso.")
            elif any(res['name'] == new_resource_name for res in app_data['resources']):
                 st.warning(f"Ya existe un recurso con el nombre '{new_resource_name}'.")
            else:
                # Crear el nuevo recurso
                derived_cost = EXPERTISE_COSTS.get(new_resource_expertise, 0)
                default_availability = get_default_availability() # Obtiene {'Lunes': 8, 'Martes': 8, ...}

                new_resource = {
                    'name': new_resource_name,
                    'expertise': new_resource_expertise,
                    'cost': derived_cost,
                    **default_availability # Desempaqueta el diccionario de disponibilidad
                }

                app_data['resources'].append(new_resource)
                st.success(f"Recurso '{new_resource_name}' ({new_resource_expertise}) añadido con disponibilidad por defecto (8h/día L-V).")
                st.rerun() # Recarga para mostrar el nuevo recurso en la tabla de abajo

    st.divider() # Separador visual

    # --- Gestión de Recursos Existentes (Editar/Eliminar/Ver Disponibilidad) ---
    st.subheader("Editar Recursos y Disponibilidad")

    # (El código del st.data_editor permanece igual que antes)
    resource_cols = ['name', 'expertise', 'cost'] + DAYS
    if app_data['resources']:
         # Asegurar que los datos están completos antes de pasar al editor
         for resource in app_data['resources']:
             resource['cost'] = EXPERTISE_COSTS.get(resource.get('expertise'), 0) # Asegurar coste correcto
             for day in DAYS:
                 resource.setdefault(day, get_default_availability().get(day, 0)) # Asegurar días

         # Ordenar recursos por nombre para consistencia
         app_data['resources'] = sorted(app_data['resources'], key=lambda x: x['name'])
         resources_df = pd.DataFrame(app_data['resources'])

         # Reordenar columnas por si acaso (aunque el sort ya las puede haber mezclado si no existían)
         resources_df = resources_df[resource_cols]
    else:
         resources_df = pd.DataFrame(columns=resource_cols)

    st.info("Usa la tabla para **editar** el nombre, expertis o las horas disponibles de los recursos existentes, o para **eliminarlos** (usando el icono de papelera al añadir/editar filas).")

    # Configuración del editor (igual que antes)
    column_config = {
        "name": st.column_config.TextColumn("Nombre Recurso", required=True),
        "expertise": st.column_config.SelectboxColumn("Expertis", options=EXPERTISE_LEVELS, required=True),
        "cost": st.column_config.NumberColumn("Coste (€/h)", format="€%d", disabled=True, help="Calculado automáticamente"), # Deshabilitado
    }
    for day in DAYS:
        column_config[day] = st.column_config.NumberColumn(
            day, min_value=0, max_value=24, step=1, format="%d h", required=True
        )

    # Usar st.data_editor para Editar/Eliminar
    edited_resources_df = st.data_editor(
        resources_df,
        key="resources_editor_v2", # Misma key que antes para mantener estado
        num_rows="dynamic", # Permite eliminar filas
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        # disabled=['cost'] # Otra forma de deshabilitar la columna coste
    )

    # --- Procesar cambios del data_editor (igual que antes) ---
    updated_resources = edited_resources_df.to_dict('records')

    final_resources = []
    resource_names_seen = set()
    ids_to_keep = set(r['name'] for r in updated_resources) # Nombres que siguen en el editor

    for resource in updated_resources:
        name = resource.get('name')
        expertise = resource.get('expertise')

        # Validar duplicados dentro de la edición actual
        if not name or name in resource_names_seen:
            # Si el nombre está vacío o duplicado EN ESTA EDICIÓN, es problemático
            # Podríamos intentar recuperar el original si es una edición, pero es complejo.
            # Por ahora, lo omitiremos si es inválido tras editar.
            st.warning(f"Edición inválida para el recurso con nombre '{name}' (vacío o duplicado en la tabla editada). No se guardarán los cambios para esta fila si son problemáticos.")
            # Buscar el recurso original para mantenerlo si la edición falla gravemente
            original_resource = next((res for res in app_data['resources'] if res['name'] == name), None)
            if original_resource and name: # Si encontramos el original y el nombre no está vacío...
                 final_resources.append(original_resource) # ...lo mantenemos sin cambios.
                 resource_names_seen.add(name)
            continue # Saltar esta fila editada

        if not expertise or expertise not in EXPERTISE_LEVELS:
             st.warning(f"Expertis inválido para el recurso '{name}'. No se guardarán cambios para esta fila.")
             original_resource = next((res for res in app_data['resources'] if res['name'] == name), None)
             if original_resource:
                 final_resources.append(original_resource)
                 resource_names_seen.add(name)
             continue

        # Validar horas diarias
        valid_hours = True
        for day in DAYS:
            hours = resource.get(day)
            if hours is None or hours < 0 or hours > 24:
                st.warning(f"Horas inválidas para {day} en el recurso '{name}'. No se guardarán cambios para esta fila.")
                valid_hours = False
                break
        if not valid_hours:
            original_resource = next((res for res in app_data['resources'] if res['name'] == name), None)
            if original_resource:
                 final_resources.append(original_resource)
                 resource_names_seen.add(name)
            continue

        # Derivar coste (se hace aquí también por si el expertis cambió)
        resource['cost'] = EXPERTISE_COSTS.get(expertise, 0)

        final_resources.append(resource)
        resource_names_seen.add(name)

    # Detectar eliminaciones: comparar nombres originales con los que quedan en el editor
    original_names = set(r['name'] for r in app_data['resources'])
    names_in_editor = set(r['name'] for r in final_resources) # Usar los nombres validados
    deleted_names = original_names - names_in_editor

    if deleted_names:
        st.success(f"Recursos eliminados: {', '.join(deleted_names)}")

    # Comparar la lista final validada con la original para ver si hubo cambios *válidos*
    # Necesitamos comparar conjuntos de diccionarios o hacer una comparación más inteligente
    # porque el orden puede cambiar por el sort o la edición.
    # Comparación simple (puede fallar si solo cambia el orden):
    if app_data['resources'] != final_resources:
         app_data['resources'] = sorted(final_resources, key=lambda x: x['name']) # Guardar ordenado
         st.success("Cambios en recursos (ediciones/eliminaciones) guardados en el estado de la sesión.")
         # st.rerun() # Evitar rerun aquí para mantener el estado del editor