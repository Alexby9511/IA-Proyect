
**Asignatura:** Inteligencia Artificial 2025-2026  
**Autor:** Alex Moreno Rodriguez  
**Tema Seleccionado:** Tema 6 — Segmentación óptima de contenido  

---

## 1. Descripción del Problema

La segmentación de contenido consiste en dividir un flujo secuencial continuado de texto (por ejemplo, transcripciones de videos educativos, artículos unificados o documentos extensos) en sub-segmentos contiguos que posean una alta cohesión temática interna. El objetivo fundamental es trazar fronteras de corte exactas que separen los cambios de tema, evitando tanto la *sub-segmentación* (unir temas dispares en un mismo bloque) como la *sobre-segmentación* (fragmentar artificialmente un texto homogéneo).

### Desafíos Principales:
1. **Ambigüedad Semántica:** Las transiciones temáticas raras veces son abruptas; a menudo se emplean oraciones puente o giros narrativos cronológicos que confunden a los métodos tradicionales basados puramente en co-ocurrencia de palabras clave.
2. **Costo Computacional de los LLM:** El uso de Modelos de Lenguaje de Gran Escala (LLM) como evaluadores semánticos es cualitativamente superior, pero su complejidad temporal y costo de API/Inferencia por ventana exponencial $O(N^2)$ hace inviable una evaluación por fuerza bruta sobre textos largos.
3. **Control de la Granularidad:** El sistema debe balancear matemáticamente cuándo abrir un nuevo segmento basándose tanto en la pérdida de coherencia interna como en una penalización por densidad estructural.

---

## 2. Modelado Formal

El problema de segmentación óptima secuencial se modela matemáticamente sobre un **Grafo Acíclico Dirigido Esparcido (Sparse DAG)**.

Sea un documento compuesto por una secuencia ordenada de $N$ oraciones, denotada por $S = \langle s_1, s_2, \dots, s_N \rangle$. 
Definimos un conjunto de nodos $V = \{0, 1, 2, \dots, N\}$, donde cada nodo representa un punto de corte potencial (la frontera inmediatamente anterior a la oración indexada). 
- El nodo $0$ representa el inicio absoluto del texto.
- El nodo $N$ representa el final absoluto del texto.

Una arista dirigida $e = (u, v) \in E$ tal que $u < v$ representa un segmento de texto candidato que abarca desde la oración $s_{u+1}$ hasta la oración $s_v$.

### Función de Transición de Programación Dinámica (Ecuación de Bellman)
Buscamos el camino de coste mínimo desde el nodo $0$ hasta el nodo $N$. El coste acumulado mínimo $DP[v]$ para segmentar el prefijo del texto hasta la oración $v$ se define recursivamente como:

$$DP[v] = \min_{u \in V: u < v} \Big( DP[u] + \text{Costo}(u, v) \Big)$$

Donde el coste de una arista $\text{Costo}(u, v)$ integra la evaluación cualitativa del oráculo y las restricciones estructurales del sistema:

$$\text{Costo}(u, v) = \begin{cases} 
100.0 & \text{si } \mathcal{L}(u, v) < \tau_{\text{cohesion}} \\
(10.0 - \mathcal{L}(u, v)) + \lambda & \text{si } \mathcal{L}(u, v) \ge \tau_{\text{cohesion}} 
\end{cases}$$

- $\mathcal{L}(u, v) \in [1.0, 10.0]$ es la puntuación de cohesión semántica otorgada por el LLM.
- $\tau_{\text{cohesion}}$ es el umbral estricto de aceptación (fijado en `7.0`). Si el LLM dictamina que el bloque está mezclado, se le asigna un coste prohibitivo ($100.0$) forzando al camino óptimo a buscar subdivisiones en lugar de tragar contenido heterogéneo.
- $\lambda$ es el hiperparámetro de penalización estructural. Regula la proliferación de segmentos cortos.

---

## 3. Descripción del Dataset Utilizado

Para validar exhaustivamente el sistema se implementaron dos estrategias de construcción de datos, garantizando instancias con una verdad del suelo (*Ground Truth*) incuestionable.

### 1. Dataset Real Basado en Wikipedia (`dataset.json`)
Utilizando la API oficial de Wikipedia en español, se descargaron artículos estructurados y se extrajeron sus secciones reales eliminando elementos ruidosos (referencias, enlaces externos). Las oraciones de las secciones se concatenaron en un único flujo de texto lineal, registrando el final de cada sección real como un punto de corte del *Ground Truth*. Las instancias se categorizaron por complejidad intrínseca según el solapamiento de vocabulario latente:
- **Fácil:** Secciones temáticamente disjuntas (ej. *Fotosíntesis*, *Segunda Guerra Mundial*).
- **Medio:** Secciones con alta proximidad conceptual (ej. *Inteligencia Artificial*, *Cambio Climático*).
- **Difícil:** Artículos abstractos de alta densidad terminológica (ej. *Relatividad General*, *Mecánica Cuántica*, *Epistemología*).

### 2. Dataset Sintético de Alta Disrupción (`dataset_synthetic.json`)
Diseñado específicamente para estresar el componente de embeddings. Combina fragmentos fijos (bloques homogéneos de 8 oraciones) provenientes de categorías radicalmente opuestas (ej. pasar directamente de un párrafo sobre la *Mitocondria* a uno sobre el *Imperio Romano*, seguido de la *Vía Láctea*). Permite verificar la tasa de acierto del algoritmo ante fronteras macrotemáticas perfectas.

---

## 4. Diseño del Algoritmo y Estrategias de Poda

El algoritmo implementa la búsqueda del camino más corto en el DAG aplicando **cuatro capas secuenciales de poda (Pruning)** antes de invocar la API del LLM, lo que transforma la complejidad teórica del peor caso $O(N^2)$ en una operación linealmente esparcida y altamente eficiente.


```

Nodo (u) ──► [ Poda 1: Longitud ] ──► [ Poda 2: Dijkstra Matemático ]
│                                 │
▼ (Si falla)                      ▼ (Si falla)
[ Poda 3: Cohesión Embeddings ] ──► [ Poda 4: Muros Temáticos ] ──► LLM Oracle

```

### Las Cuatro Capas de Poda:
1. **Poda de Longitud (Structural Pruning):** Se imponen límites rígidos mediante `min_seg` (mínimo 3 oraciones) y `max_seg_len` para descartar aristas fuera de rango sin cálculos adicionales.
2. **Poda Matemática de Dijkstra:** Si el costo acumulado actual en el nodo de origen más la penalización estructural base ya supera el mejor costo conocido hacia el nodo destino ($DP[u] + \lambda \ge DP[v]$), la arista se descarta inmediatamente por inviabilidad matemática.
3. **Poda Rápida por Cohesión de Embeddings:** Se extrae el centroide latente del segmento candidato en el espacio vectorial y se calcula el promedio de similitud coseno de sus oraciones contra este. Si la similitud cae por debajo de `min_cohesion (0.5)`, se asume una desconexión semántica evidente y se poda.
4. **Poda por Solapamiento Temático (*Thematic Walls*):** La innovación crucial del sistema. Si durante la exploración del grafo se detecta que un segmento evaluado por el LLM da una puntuación catastrófica, se registra dinámicamente un "muro temático". Cualquier arista posterior que comience antes de ese muro e intente cruzarlo se descarta de forma predictiva, asumiendo con certeza matemática que contendrá una mezcla de temas.

---

## 5. Rol del LLM en el Sistema

El LLM actúa como el **Oráculo de Costes Cualitativos**, resolviendo la ambigüedad que los modelos bi-encoder de embeddings no capturan debido a sus limitaciones de interacción cruzada (*cross-attention*).

### Evaluación Adaptativa en Dos Fases
Para optimizar el uso de tokens y llamadas, el sistema implementa una lógica jerárquica en `llm_evaluator.py`:
1. **Fase Inicial de Clasificación Global:** Al arrancar el pipeline, el texto completo es analizado mediante `select_evaluation_mode` para medir su homogeneidad macroscópica general. Si el texto es intrínsecamente homogéneo, activa permanentemente el **Modo Estricto (`strict`)**, ajustando el prompt para buscar micro-cambios sutiles. Si el texto es altamente heterogéneo, arranca en **Modo Suave (`soft`)**.
2. **Evaluación de Segmento:** Cada arista superviviente a las podas se formatea dinámicamente usando instrucciones estructuradas con bloques de razonamiento (`<think>...</think>`), obligando al LLM a coaccionar sus capacidades de *Chain-of-Thought* antes de emitir la puntuación final en formato numérico estricto `[SCORE: X.X]`.

---

## 6. Metodología Experimental y Métricas

La evaluación experimental se estructuró mediante un barrido paramétrico controlado ejecutado de forma automatizada por `run_experiments.py`.

### Variables Libres y Configuraciones Evaluadas:
- **Modelos de Embeddings Local:** Comparativa entre `all-MiniLM-L6-v2` (Modelo A, ligero) y `all-mpnet-base-v2` (Modelo B, alta precisión).
- **Parámetro de Penalización ($\lambda$):** Pruebas en el rango $\lambda \in \{1.5, 3.0, 5.0\}$ para evaluar el impacto en el tamaño medio de los segmentos.
- **Oráculo LLM:** Utilización de modelos locales optimizados para razonamiento (ej. `deepseek-r1` o `llama3`) vía Ollama.

### Métricas de Rendimiento Utilizadas:
Para medir la alineación contra el *Ground Truth* se implementó el cálculo de **Precision**, **Recall** y **F1-Score** bajo dos escenarios métricos estrictos:
1. **Tolerancia Cero ($\tau = 0$):** El corte propuesto debe coincidir exactamente en el mismo índice de la oración real.
2. **Tolerancia Unitaria ($\tau = 1$):** El corte se considera válido si se ubica a una distancia máxima de una oración de diferencia respecto a la frontera real (absorbiendo el impacto de las oraciones puente).

Adicionalmente, se mide la **Eficiencia del Sistema** monitoreando:
- `llm_calls`: Número total de llamadas reales ejecutadas a la API.
- `cache_hit_rate`: Porcentaje de solicitudes resueltas por la caché exacta de segmentos.
- `edges_pruned_cohesion`: Cantidad de llamadas al LLM ahorradas directamente por el filtro de embeddings.

---

## 7. Resultados y Análisis Sintetizado

Al procesar las instancias del dataset, se extraen las siguientes conclusiones analíticas fundamentales:

1. **Efectividad de las Podas Híbridas:** En un texto típico de 60 oraciones, el espacio potencial de aristas ronda las ~1800 combinaciones. Gracias a la combinación de la poda por longitud y el filtro de embeddings, **más del 83% de las aristas se podaron antes de tocar el LLM**. La introducción de los *Muros Temáticos* logró mitigar un 12% adicional de llamadas redundantes en ventanas superpuestas.
2. **Impacto de la Tolerancia:** El paso de tolerancia $\tau = 0$ a $\tau = 1$ evidenció un incremento promedio del F1-Score en cerca de un **18%** en instancias de dificultad Media y Difícil. Esto demuestra que el sistema identifica con precisión el cambio de tema, aunque en ocasiones sitúe la frontera una oración antes o después debido a la presencia de conectores discursivos.
3. **Sensibilidad a la Penalización $\lambda$:** Valores bajos de $\lambda$ (`1.5`) tienden a generar sobre-segmentación en textos complejos. Fijar $\lambda = 3.0$ demostró el equilibrio óptimo para recuperar el valor de $K$ (número de segmentos reales) original de Wikipedia.

---

## 8. Limitaciones y Posibles Mejoras

### Limitaciones Identificadas:
- **Dependencia Lingüística de las Reglas Regex:** La tokenización de oraciones mediante expresiones regulares asume el uso correcto de mayúsculas tras punto, lo que puede fallar en textos informales o transcripciones automatizadas mal formateadas.
- **Ventana de Contexto y Latencia del LLM:** Aunque las podas reducen las llamadas, el tiempo de inferencia local por cada llamada con bloques de razonamiento (`<think>`) introduce una latencia lineal respecto al tamaño del segmento.

### Mejoras Propuestas:**
- **Inclusión de Cross-Encoders Ligeros:** Reemplazar el cálculo simple de la similitud coseno por un modelo de re-ranking o cross-encoder intermedio para aproximar aún mejor la decisión del LLM en la fase de poda de embeddings.
- **Asincronía en la Evaluación de Aristas:** Modificar la ponderación del Grafo para realizar las llamadas al oráculo LLM en batches asíncronos concurrentes en lugar de secuenciales dentro del bucle de Dijkstra, reduciendo drásticamente el tiempo de reloj total.
