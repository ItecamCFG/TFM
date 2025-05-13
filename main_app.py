# main_app.py
import streamlit as st

# Importar módulos de UI y gestión de datos (usando v2)
from data_manager import initialize_session_state, save_data, load_data, DATA_FILE
from ui_projects import display_project_management
from ui_resources import display_resource_management
from ui_planning import display_planning

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Gestor de Proyectos ", # Nuevo título
    page_icon="📊",
    layout="wide"
)

st.title("🏢📊📈 Planificador Inteligente.")
st.caption("Desarrollado por Itecam-Centro Tecnológico Industrial.")

# --- Inicializar Estado (Usará data_manager v2) ---
initialize_session_state()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Opciones")
    st.subheader("Gestión de Datos")
    if st.button("💾 Guardar Datos Actuales"):
        save_data(st.session_state.app_data)

    uploaded_file = st.file_uploader("📂 Cargar Datos desde Archivo", type=["json"], key="file_uploader_v2")
    if uploaded_file is not None:
        try:
             # Guardar el archivo subido sobre el DATA_FILE actual
             with open(DATA_FILE, "wb") as f:
                f.write(uploaded_file.getbuffer())
             # Forzar recarga de datos en el estado
             loaded_data = load_data()
             # Convertir deadlines de string a date al cargar (importante hacerlo aquí también)
             for project in loaded_data.get('projects', []):
                if isinstance(project.get('deadline'), str):
                    try:
                        project['deadline'] = date.fromisoformat(project['deadline'])
                    except (ValueError, TypeError):
                        project['deadline'] = None
             st.session_state.app_data = loaded_data
             st.success("Archivo cargado y datos actualizados.")
             # Limpiar el uploader para permitir recargar el mismo archivo
             # st.session_state.file_uploader_v2 = None # Esto a veces causa problemas, probar con cuidado
             st.rerun()
        except Exception as e:
             st.error(f"Error al procesar el archivo subido: {e}")


    st.divider()
    st.info(f"Datos guardados/cargados: `{DATA_FILE}`")
    st.caption("Recuerda guardar los cambios importantes.")

# --- Área Principal con Pestañas ---
tab1, tab2, tab3 = st.tabs([
    " Proyectos y Tareas Secuenciales ",
    " Recursos y Disponibilidad ",
    " Planificación Optimizada "
])

with tab1:
    display_project_management()

with tab2:
    display_resource_management()

with tab3:
    display_planning()

# --- Pie de página o información adicional (Opcional) ---
st.divider()
st.markdown("---")
st.caption("Modelo V2: Incluye secuencias, deadlines y expertis.")