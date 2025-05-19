import streamlit as st
from optimization.data_models import OptimizationInput

def generate_model_diagnostics(model):
    """Genera una tabla de resumen del modelo de optimización, adaptada según el tipo."""
    st.markdown("🔍 **Detalle del modelo de optimización:**")

    try:
        # Resumen general
        st.markdown(f"- Horizonte de planificación: **{len(model.days_list)} días**")
        st.markdown(f"- Recursos: **{len(model.resource_names)}**")
        st.markdown(f"- Proyectos: **{len(model.project_names)}**")
        total_tasks = sum(len(v) for v in model.tasks_per_project_name.values())
        st.markdown(f"- Tareas totales: **{total_tasks}**")

        st.markdown("### Variables del modelo")

        if hasattr(model, "variables") and isinstance(model.variables, dict) and "x" in model.variables:
            def count_nested(d):
                return sum(len(v) for v in d.values()) if isinstance(d, dict) else 0

            x = len(model.variables.get("x", {}))
            work_day = len(model.variables.get("work_day", {}))

            # Detectar si es QUBO (usa bits) o MILP (usa variables continuas/enteras)
            if "y_bits" in model.variables:
                # Caso QUBO
                y = count_nested(model.variables.get("y_bits", {}))
                end_day = count_nested(model.variables.get("end_day_bits", {}))
                makespan = len(model.variables.get("makespan_bits", {}))
                st.markdown(f"- 🔹 Asignación `x`: **{x}**")
                st.markdown(f"- 🔹 Trabajo diario `y` (bits): **{y}**")
                st.markdown(f"- 🔹 Días de trabajo `work_day`: **{work_day}**")
                st.markdown(f"- 🔹 Finalización `end_day` (bits): **{end_day}**")
                st.markdown(f"- 🔹 Makespan (bits): **{makespan}**")
            else:
                # Caso MILP
                y = len(model.variables.get("y", {}))
                end_day = len(model.variables.get("end_day", {}))
                makespan = 1 if "makespan" in model.variables else 0
                st.markdown(f"- 🔹 Asignación `x`: **{x}**")
                st.markdown(f"- 🔹 Trabajo diario `y`: **{y}**")
                st.markdown(f"- 🔹 Días de trabajo `work_day`: **{work_day}**")
                st.markdown(f"- 🔹 Finalización `end_day`: **{end_day}**")
                st.markdown(f"- 🔹 Makespan: **{makespan}**")

            # Slacks (opcionales)
            #slack_keys = [k for k in model.variables if "slack" in k.lower()]
            #slack_total = sum(count_nested(model.variables[k]) for k in slack_keys)
            #st.markdown(f"- ⚙️ Variables auxiliares (slacks): **{slack_total}**")

        elif hasattr(model, "model"):
            try:
                solver_model = model.model

                # --- PuLP ---
                if hasattr(solver_model, "variables") and callable(solver_model.variables):
                    num_vars = len(solver_model.variables())
                    num_cons = len(solver_model.constraints)
                    st.markdown(f"- 🔹 Variables totales (PuLP): **{num_vars}**")
                    st.markdown(f"- 🔹 Restricciones totales (PuLP): **{num_cons}**")

                # --- PySCIPOpt ---
                elif hasattr(solver_model, "getVars") and hasattr(solver_model, "getConss"):
                    num_vars = len(solver_model.getVars())
                    num_cons = len(solver_model.getConss())
                    st.markdown(f"- 🔹 Variables totales (SCIP): **{num_vars}**")
                    st.markdown(f"- 🔹 Restricciones totales (SCIP): **{num_cons}**")

                else:
                    st.info("Modelo MILP detectado pero no se reconoció el solver específico.")
            except Exception as e:
                st.warning(f"No se pudieron contar variables o restricciones: {e}")

        else:
            st.info("Este modelo aún no ha definido su estructura interna.")

    except Exception as e:
        st.warning(f"Error al generar diagnóstico del modelo: {e}")


def generate_input_summary(input_data: OptimizationInput):
    st.markdown("🧾 **Resumen de entrada:**")
    st.markdown(f"- Proyectos: **{len(input_data.projects)}**")
    st.markdown(f"- Tareas: **{len(input_data.tasks)}**")
    st.markdown(f"- Recursos: **{len(input_data.resources)}**")
