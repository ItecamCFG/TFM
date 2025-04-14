# TFM: Prototipo de Planificación de Proyectos

Este repositorio contiene un **primer prototipo** para la resolución de un problema de planificación de proyectos (**PPP**, *Project Planning Problem*). La idea base se inspira en el **Job Scheduling Problem** (**JSP**), y sirve como **demo** de las capacidades de **Itecam** en la **resolución de problemas de optimización combinatoria**. Además, este trabajo busca sentar las bases para explorar diferentes técnicas de optimización dentro de un entorno **B2B**.

---

## Motivación

En muchas organizaciones, la **planificación de proyectos** implica asignar recursos y programar tareas, respetando restricciones de tiempo, especialización del personal y costos. El objetivo común suele ser **minimizar** el coste total o el tiempo total de finalización, al tiempo que se **maximizan** la eficiencia y la satisfacción de las fechas límite.

El **Project Planning Problem** (PPP) propuesto aquí es una **variante** del Job Scheduling Problem que aborda:

- **Asignación de tareas** a recursos específicos (personas, equipos).
- **Restricciones** de capacidad y disponibilidad.
- **Deadlines** y posibles secuencias entre tareas.
- **Optimización** de costes y/o tiempos.

---

## Estructura del Repositorio

1. **`model_debug.py`**  
   Contiene la lógica de modelado y la definición de dos funciones principales de resolución:
   - `solve_scheduling_problem()` (usando [PuLP](https://github.com/coin-or/pulp) con CBC).
   - `solve_scheduling_problem_scip()` (usando [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) con SCIP).  
   Ambas comparten la misma filosofía de modelado pero difieren en el *solver*.

2. **`main_app.py`**  
   Un script de ejemplo que muestra cómo:
   - Cargar datos reales (almacenados en JSON).
   - Preparar estructuras (proyectos, tareas, recursos, etc.).
   - Invocar las funciones de optimización.
   - Imprimir los resultados y asignaciones.

3. **`app_data_v2.json`**  
   Un fichero de prueba con la **configuración base** de proyectos, tareas y recursos, junto a sus parámetros (costes, disponibilidad diaria, etc.). Permite **validar** el prototipo de planificación.

4. **Documentación** y archivos misceláneos  
   Este repositorio también puede incluir notebooks, documentación extendida o archivos de configuración, ofreciendo un registro más detallado de la evolución del prototipo.

---

## Cómo Empezar

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/tfm-planning.git
   cd tfm-planning
   ```

2. **Instalar dependencias** (puedes usar un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```
   Los requisitos mínimos incluyen:
   - [PuLP](https://pypi.org/project/PuLP/)
   - [PySCIPOpt](https://pypi.org/project/pyscipopt/)
   - Otros paquetes auxiliares (p.ej. `pandas` si fuera necesario).

3. **Ejecutar un ejemplo**:
   - `python main_app.py`  
     Carga el fichero `app_data_v2.json`, define un horizonte de planificación y llama a **CBC** y **SCIP**, mostrando el coste y las asignaciones calculadas.

4. **Probar otras configuraciones**:
   - Ajustar parámetros en `app_data_v2.json` (ej. cambiar horas, recursos, etc.).
   - Modificar el **horizonte de planificación** (número de días) en `main_app.py`.
   - Explorar mejoras en `model_debug.py` para **incluir** restricciones adicionales (p.ej. secuencias estrictas, dependencia entre tareas, preferencias de recursos, etc.).

---

## Perspectivas Futuras

Este prototipo es una **prueba de concepto** que se puede expandir y perfeccionar. Algunas líneas futuras incluyen:

- **Incorporar secuencias y *deadlines*** en la formulación matemática de manera explícita.  
- **Diversificar métricas** de optimización (tiempo total vs. coste total vs. tardanza).  
- **Experimentar** con otros solvers (Gurobi, CPLEX, GLPK, etc.) y comparar rendimiento.  
- **Desplegar una app web** con [Streamlit](https://streamlit.io/), permitiendo a usuarios no técnicos configurar y lanzar optimizaciones con un clic.  
- **Integrar heurísticas** o metaheurísticas (e.g., algoritmos genéticos, búsqueda tabú) cuando la escala de datos se vuelve muy grande para la optimización exacta.

---

## Contribuciones

¡Bienvenidas toda clase de **ideas y sugerencias**! Por favor, abre un *issue* o un *pull request* si deseas colaborar.

---

## Licencia

Este proyecto se publica bajo [MIT License](LICENSE) (o la que corresponda). Revisa el archivo de licencia para más detalles.

--- 

### Contacto

Para dudas o más detalles sobre la integración con Itecam, contacta con:

- **Itecam** - Centro Tecnológico Industrial de Castilla-La Mancha  
- Correo: [iteligencia.computacion@itecam.com](mailto:transformacion.digital@itecam.com)

¡Gracias por tu interés en el proyecto!

