# TFM: Planificador Inteligente - Solución RCPSP Multialgoritmo

Este repositorio contiene una **plataforma avanzada de investigación** para la resolución del **Resource-Constrained Project Scheduling Problem (RCPSP)** utilizando múltiples paradigmas de optimización. Desarrollado por **Itecam - Centro Tecnológico Industrial de Castilla-La Mancha** como demostración de capacidades en **optimización combinatoria**, **computación cuántica** y **metaheurísticas**.

---

## 🎯 Problema: RCPSP (Resource-Constrained Project Scheduling Problem)

El **RCPSP** es un problema clásico de optimización combinatoria que busca programar un conjunto de tareas con restricciones de recursos, precedencias y deadlines. Nuestro enfoque aborda:

### **Características del Problema:**
- **Proyectos múltiples** con tareas interdependientes
- **Recursos limitados** con diferentes niveles de expertise (Junior, Senior, Experto)
- **Restricciones temporales**: secuencias obligatorias entre tareas
- **Disponibilidad variable** de recursos por día de la semana
- **Deadlines** por proyecto con penalizaciones por retraso
- **Objetivo**: Minimizar el **makespan** (tiempo total de finalización)

### **Formulación Matemática:**
- Variables de asignación binarias: `x[p,t,r]` (tarea t del proyecto p asignada al recurso r)
- Variables de horas: `y[p,t,r,d]` (horas trabajadas por recurso r en tarea t del proyecto p el día d)
- Restricciones de expertise, disponibilidad, secuencialidad y deadlines
- Función objetivo: `min(makespan)` donde `makespan ≥ max(end_day[p,t])`

---

## 🔬 Algoritmos de Optimización Implementados

La plataforma incluye múltiples enfoques algorítmicos para comparar rendimiento y explorar diferentes paradigmas:

### **1. Optimización Exacta**
- **PuLP + CBC** (`MakespanMinimizationPuLP`): Programación lineal entera con solver CBC
- **SCIP** (`MakespanMinimizationSCIP`): Solver de optimización combinatoria avanzado

### **2. Computación Cuántica**
- **Neal Simulated Annealing** (`NealMakespanModel`): QUBO usando recocido simulado cuántico
- **D-Wave Hybrid** (`DWaveHybridMakespanModel`): Computación cuántica híbrida clásica-cuántica

### **3. Metaheurísticas**
- **Algoritmo Genético** (`GeneticMakespanPlanner`): Enfoque evolutivo con cruce y mutación especializados

### **4. Arquitectura Modular**
- **Clase base** `OptimizationModel`: Patrón Strategy para fácil extensión
- **Modelos de datos** estructurados con `dataclasses`
- **Diagnósticos** automáticos de modelos y validación de restricciones

---

## 🏗️ Estructura del Repositorio

```
TFM/
├── 📱 Interfaz de Usuario (Streamlit)
│   ├── main_app.py              # Aplicación principal
│   ├── ui_projects.py           # Gestión de proyectos
│   ├── ui_resources.py          # Gestión de recursos
│   ├── ui_planning.py           # Interfaz de planificación
│   └── ui_dwave.py              # Laboratorio QUBO
│
├── 🔧 Motor de Optimización
│   ├── optimization/
│   │   ├── base_model.py        # Clase base abstracta
│   │   ├── data_models.py       # Estructuras de datos
│   │   ├── model_pulp.py        # Solver PuLP/CBC
│   │   ├── model_SCIP.py        # Solver SCIP
│   │   ├── neal_model.py        # Simulated Annealing
│   │   ├── dwave_hybrid_model.py # D-Wave Hybrid
│   │   ├── genetic_model.py     # Algoritmo Genético
│   │   └── model_diagnostics.py # Diagnósticos
│
├── 📊 Visualización y Análisis
│   ├── visualization/
│   │   └── aux_visualizations.py # Gantt charts, métricas
│   ├── utils.py                 # Logging experimental
│   └── solver_benchmark_results.csv # Resultados benchmarks
│
├── 💾 Gestión de Datos
│   ├── data_manager.py          # CRUD datos aplicación
│   ├── app_data_v2.json         # Datos de prueba
│   └── requirements.txt         # Dependencias
│
└── 📋 Documentación
    ├── README.md                # Este archivo
    └── genetic_algorithm_issues.md # Mejoras propuestas
```

---

## 🚀 Instalación y Uso

### **Prerequisitos**
- Python 3.12+
- Entorno virtual recomendado

### **1. Clonar e Instalar**
```bash
git clone https://github.com/tu-usuario/tfm-planning.git
cd tfm-planning

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### **2. Ejecutar la Aplicación**
```bash
streamlit run main_app.py
```

### **3. Usar la Interfaz Web**
1. **Proyectos y Tareas**: Definir proyectos con tareas secuenciales
2. **Recursos**: Configurar personas con expertise y disponibilidad
3. **Planificación**: Seleccionar algoritmo y ejecutar optimización
4. **Laboratorio QUBO**: Experimentar con parámetros cuánticos

---

## 📊 Características Principales

### **🎛️ Interfaz Streamlit Avanzada**
- Dashboard interactivo con 4 pestañas especializadas
- Visualizaciones Gantt dinámicas con Plotly
- Carga/descarga de datos JSON/Excel
- Comparación de algoritmos en tiempo real

### **🔍 Sistema de Benchmarking**
- Logging automático de experimentos
- Métricas de rendimiento: tiempo, makespan, factibilidad
- Validación de restricciones post-optimización
- Histórico de resultados en CSV

### **⚡ Optimización Robusta**
- Manejo de problemas infeasibles
- Timeouts configurables
- Validación exhaustiva de soluciones
- Diagnósticos automáticos de modelos

### **🔬 Experimentación Cuántica**
- Formulación QUBO con penalizaciones adaptativas
- Análisis de energías por tipo de restricción
- Comparación clásico vs. cuántico
- Parámetros ajustables para hardware cuántico

---

## 📈 Resultados y Benchmarks

### **Rendimiento por Algoritmo** (Datos de ejemplo)
| Algoritmo | Tiempo (s) | Makespan | Estado | Observaciones |
|-----------|------------|----------|--------|---------------|
| PuLP/CBC | 0.43 | 3.0 | Optimal | Más confiable para problemas pequeños |
| SCIP | 0.38 | 3.0 | Optimal | Mejor rendimiento, soporte escalable |
| Neal QUBO | 2.18 | 5.0 | Infeasible* | Requiere ajuste de penalizaciones |
| D-Wave Hybrid | - | - | En desarrollo | Para problemas grandes |
| Genético | 15.2 | 3.2 | Feasible | Buena aproximación, escalable |

*\*Los modelos QUBO están en proceso de calibración de parámetros*

### **Casos de Uso Validados**
- **Proyecto Fotovoltaico**: 2 tareas secuenciales, 8-10 horas cada una
- **Chatbot PRL**: 1 tarea de 6 horas, expertise Senior
- **Horizonte**: 10-30 días laborables
- **Recursos**: 2 personas (Junior/Senior), disponibilidad 8h/día L-V

---

## 🔧 Dependencias Principales

```python
# Optimización Clásica
PuLP==3.1.1                    # Programación lineal
PySCIPOpt==5.5.0              # Solver SCIP

# Computación Cuántica
dwave-ocean-sdk==8.4.0        # Suite completa D-Wave
dwave-neal==0.6.0             # Simulated Annealing
dwave-hybrid==0.6.14          # Algoritmos híbridos

# Interfaz y Visualización
streamlit==1.45.1             # Aplicación web
plotly>=5.0.0                 # Gráficos interactivos
pandas>=2.0.0                 # Manipulación de datos

# Cálculo Científico
numpy>=1.24.0                 # Computación numérica
scipy>=1.10.0                 # Algoritmos científicos
```

---

## 🚧 Trabajo Futuro

### **Mejoras Inmediatas**
- [ ] **Calibración QUBO**: Ajustar penalizaciones para Neal model
- [ ] **Validación robusta**: Mejorar detección de infeasibilidad
- [ ] **Paralelización**: GPU acceleration para algoritmo genético
- [ ] **Multi-objetivo**: Makespan + costo simultáneamente

### **Funcionalidades Avanzadas**
- [ ] **Planificación robusta**: Manejo de incertidumbre en duraciones
- [ ] **Dependencias complejas**: Más allá de secuencias simples
- [ ] **Recursos compartidos**: Setup times y restricciones de ubicación
- [ ] **Machine Learning**: Predicción de duraciones basada en históricos

### **Investigación**
- [ ] **Quantum Advantage**: Análisis comparativo en hardware real
- [ ] **Formulaciones híbridas**: Combinación de paradigmas
- [ ] **Escalabilidad**: Problemas industriales grandes (>1000 tareas)
- [ ] **Optimización en tiempo real**: Re-planificación dinámica

---

## 🤝 Contribuciones

Este proyecto está abierto a contribuciones de la comunidad científica y técnica:

### **Cómo Contribuir**
1. **Fork** el repositorio
2. **Crear branch** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** tus cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. **Push** al branch: `git push origin feature/nueva-funcionalidad`
5. **Crear Pull Request**

### **Áreas de Contribución**
- 🔬 **Algoritmos**: Nuevos solvers o mejoras existentes
- 🎨 **UI/UX**: Mejoras en la interfaz Streamlit
- 📊 **Visualización**: Nuevos tipos de gráficos o dashboards
- 🧪 **Testing**: Casos de prueba y benchmarks
- 📚 **Documentación**: Tutoriales y ejemplos

---

## 📄 Licencia

Este proyecto se distribuye bajo la **MIT License**. Ver [LICENSE](LICENSE) para más detalles.

---

## 👥 Contacto y Colaboración

**Desarrollado por:**
- **Itecam** - Centro Tecnológico Industrial de Castilla-La Mancha
- **Email**: [inteligencia.computacion@itecam.com](mailto:inteligencia.computacion@itecam.com)
- **Web**: [www.itecam.com](https://www.itecam.com)

**Colaboraciones académicas y empresariales bienvenidas.**

---

## 🙏 Agradecimientos

- **D-Wave Systems** por el acceso a tecnología cuántica
- **Comunidad Open Source** por las herramientas de optimización
- **Streamlit** por la plataforma de desarrollo web
- **Equipos de investigación** en optimización combinatoria

---

*¿Interesado en optimización combinatoria o computación cuántica? ¡Contáctanos para explorar colaboraciones!*

