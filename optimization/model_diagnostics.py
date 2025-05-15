import streamlit as st

def generate_model_diagnostics(model):
    """Genera una tabla de resumen del modelo de optimización, adaptada según el tipo."""
    st.markdown("🔍 **Detalle del modelo de optimización:**")

    try:
        st.markdown(f"- Horizonte de planificación: **{len(model.days_list)} días**")
        st.markdown(f"- Recursos: **{len(model.resource_names)}**")
        st.markdown(f"- Proyectos: **{len(model.project_names)}**")
        total_tasks = sum(len(v) for v in model.tasks_per_project_name.values())
        st.markdown(f"- Tareas totales: **{total_tasks}**")

        st.markdown("### Variables del modelo")

        # Caso QUBO (usa self.variables binario)
        if hasattr(model, "variables") and "x" in model.variables:
            def count_nested(d):
                return sum(len(v) for v in d.values()) if isinstance(d, dict) else 0

            x = len(model.variables.get("x", {}))
            y = count_nested(model.variables.get("y_bits", {}))
            work_day = len(model.variables.get("work_day", {}))
            end_day = count_nested(model.variables.get("end_day_bits", {}))
            makespan = len(model.variables.get("makespan_bits", {}))

            st.markdown(f"- 🔹 Asignación `x`: **{x}**")
            st.markdown(f"- 🔹 Trabajo diario `y` (bits): **{y}**")
            st.markdown(f"- 🔹 Días de trabajo `work_day`: **{work_day}**")
            st.markdown(f"- 🔹 Finalización `end_day` (bits): **{end_day}**")
            st.markdown(f"- 🔹 Makespan (bits): **{makespan}**")

            slack_keys = [k for k in model.variables if "slack" in k.lower()]
            slack_total = sum(count_nested(model.variables[k]) for k in slack_keys)
            st.markdown(f"- ⚙️ Variables auxiliares (slacks): **{slack_total}**")

        # Caso MILP: PuLP o PySCIPOpt
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
