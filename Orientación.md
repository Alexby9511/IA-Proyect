# Orientación del Proyecto Final Inteligencia Artificial 2025-2026

Desarrollar un sistema que resuelva un problema de optimización, planificación, satisfacción de restricciones, ... mediante técnicas y herramientas dadas en la asignatura, incorporando un modelo de lenguaje (LLM) como parte funcional del sistema.

Cada equipo (equipos de 1 persona) deberá seleccionar uno de los 10 temas propuestos:

| Tema | Descripción |
|------|-------------|
| 1 | **Priorización de casos clínicos bajo recursos limitados**<br>Diseñar un sistema que, dado un conjunto de pacientes con información estructurada y descripciones textuales, determine un orden de atención que maximice el impacto total bajo restricciones de recursos. El sistema debe utilizar un LLM para interpretar o enriquecer la información textual de los casos y apoyar la evaluación de las decisiones. |
| 2 | **Construcción de resúmenes como selección óptima de segmentos**<br>Dado un conjunto de fragmentos extraídos de un video educativo, seleccionar y ordenar un subconjunto que represente adecuadamente el contenido bajo un límite de longitud. El sistema debe utilizar un LLM para evaluar la coherencia y relevancia de los resúmenes generados. |
| 3 | **Planificación de trayectoria profesional**<br>Dado un conjunto de habilidades, cursos y restricciones de prerequisitos, construir una secuencia válida que permita alcanzar un objetivo profesional. El sistema debe utilizar un LLM para interpretar objetivos en lenguaje natural y evaluar la calidad de las trayectorias propuestas. |
| 4 | **Detección de patrones sospechosos en secuencias de mensajes**<br>Dada una secuencia de mensajes de una conversación, identificar subsecuencias que correspondan a comportamientos sospechosos definidos por reglas generales. El sistema debe utilizar un LLM para analizar el contenido textual y apoyar la interpretación de patrones complejos. |
| 5 | **Reparación de inconsistencias en historiales**<br>Dado un conjunto de hechos (estructurados y textuales) que pueden contener contradicciones, encontrar un conjunto mínimo de modificaciones que restaure la coherencia global. El sistema debe utilizar un LLM para evaluar la consistencia semántica entre los elementos del historial. |
| 6 | **Segmentación óptima de contenido**<br>Dada una secuencia de elementos (texto o fragmentos de contenido), dividirla en segmentos contiguos que maximicen la coherencia interna. El sistema debe utilizar un LLM para evaluar la cohesión semántica de cada segmento. |
| 7 | **Exploración de trayectorias profesionales alternativas**<br>Dado un conjunto de decisiones posibles en una carrera profesional, generar y evaluar múltiples trayectorias válidas bajo distintos criterios. El sistema debe utilizar un LLM para analizar y comparar cualitativamente las trayectorias generadas. |
| 8 | **Generación de mensajes bajo restricciones estructurales**<br>Construir mensajes que cumplan un conjunto de reglas estructurales y restricciones definidas (longitud, contenido, forma). El sistema debe utilizar un LLM como componente en la generación y evaluación de los mensajes. |
| 9 | **Construcción de rutas de aprendizaje**<br>Dado un conjunto de recursos con dependencias, construir una secuencia válida que optimice el aprendizaje bajo restricciones de tiempo. El sistema debe utilizar un LLM para interpretar descripciones de recursos y evaluar su utilidad relativa. |
| 10 | **Identificación de anomalías en secuencias**<br>Dada una secuencia de eventos, encontrar el subconjunto mínimo cuya eliminación permite que la secuencia cumpla un conjunto de reglas. El sistema debe utilizar un LLM para interpretar el significado de los eventos y apoyar la evaluación de las reglas. |

## Diseño experimental

El proyecto debe incluir:

- Conjunto de instancias de prueba (generadas o reales) (generadas u obtenidas por el estudiante)
- Comparación entre distintas configuraciones o variantes
- Análisis del comportamiento del sistema

## Profe, ¿qué debo entregar?

### 1. Código fuente

Debe incluir:
- Dataset
- Implementación completa del sistema
- Instrucciones de ejecución
- Configuración del uso del LLM

### 2. Informe técnico

Debe incluir:
- Descripción del problema
- Modelado formal
- Descripción del dataset utilizado, ya sea obtenido o generado y, en caso de ser generado, describir el proceso de generación y explicar por qué cumple las condiciones para el problema definido
- Diseño del algoritmo
- Rol del LLM en el sistema
- Metodología experimental
- Resultados y análisis
- Limitaciones y posibles mejoras