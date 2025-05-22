# Mejoras Propuestas para el Algoritmo Genético

Este archivo resume posibles *issues* a crear para ir mejorando el `GeneticMakespanPlanner` y escalar su rendimiento.

## Ideas de Optimización y Escalabilidad

1. **Validación y corrección de decodificador**
   - Revisar que `_evaluate_and_decode` produzca calendarios factibles.
   - Añadir tests unitarios que detecten inconsistencias en la asignación.
2. **Uso de GPU para la evaluación**
   - Investigar bibliotecas como CuPy o numba para paralelizar en GPU los bucles críticos de la evaluación de individuos.
   - Crear interfaces que permitan seleccionar backend CPU/GPU.
3. **Estrategias de selección y cruce avanzadas**
   - Implementar métodos alternativos (torneos adaptativos, selección por rango, etc.).
   - Evaluar cruces específicos para problemas de secuenciación.
4. **Paralelización de poblaciones**
   - Dividir la población en subpoblaciones que se evalúen de forma independiente (islas) y se mezclen periódicamente.
5. **Ajuste dinámico de parámetros**
   - Adaptar probabilidades de mutación y cruce según la convergencia observada.
6. **Persistencia y reinicio**
   - Guardar el estado de las mejores soluciones para reanudar ejecuciones largas.
7. **Integración con `OptimizationConfig`**
   - Exponer todos los parámetros del GA en la configuración para facilitar pruebas.

Estas mejoras podrán abordarse de manera gradual en futuros *pull requests*.
