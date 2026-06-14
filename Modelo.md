1. El Modelo Matemático (El Núcleo del Sistema)
Vamos a modelar el texto como un Grafo Acíclico Dirigido (DAG) .
Si tenemos un texto con N oraciones, creamos N+1 nodos, donde cada nodo representa un "punto de corte" potencial entre oraciones (el nodo 0 es el inicio del texto, el nodo N es el final).
Una arista dirigida desde el nodo i hasta el nodo j (donde i<j) representa un segmento de texto que va desde la oración i hasta la oración j. Nuestro objetivo es encontrar el camino de menor coste desde 0 hasta N. Esta formulación de Programación Dinámica es matemáticamente idéntica a buscar la distancia de conexión óptima, por lo que puedes resolver la matriz usando el algoritmo de Dijkstra.
La ecuación de transición de nuestra Programación Dinámica será:
DP[j]=0≤i<j
min
(DP[i]+Costo(i,j)+λ)
Donde:
DP[j] es el coste mínimo para segmentar óptimamente las primeras j oraciones.
Costo(i,j) es la "penalización por falta de coherencia" del segmento que va de i a j. Un segmento muy coherente tendrá un coste cercano a 0.
λ es un hiperparámetro de penalización estructural. Evita que el algoritmo haga trampa creando un segmento por cada oración (lo cual tendría coherencia perfecta, pero sería inútil).
2. El Rol del LLM (El Oráculo de Costes)
El mayor peligro de la Programación Dinámica es la complejidad computacional. Si evaluamos todos los posibles segmentos (i,j) con el LLM, tendríamos O(N2
) llamadas al modelo, lo cual paralizaría tu entorno local en Ubuntu.
Para solucionarlo, usaremos una arquitectura de dos fases para la función Costo(i,j):
Filtro Rápido (Embeddings): Usamos los modelos locales (como paraphrase-multilingual-mpnet-base-v2) para descartar aristas inválidas. Si la similitud coseno interna de un segmento (i,j) cae por debajo de un umbral, asumimos que el coste es infinito (∞) y podamos esa rama del grafo sin llamar al LLM. Además, limitamos la longitud máxima de un segmento a un número razonable de oraciones.
Evaluación Precisa (LLM): Para los segmentos candidatos que sobreviven al filtro, le pasamos el texto al LLM con un prompt de clasificación estricto. Por ejemplo: "¿Este bloque de texto mantiene un único tema central sin desviaciones? Responde 1 (Sí) o 0 (No)". Transformamos esa respuesta binaria (o una escala muy acotada) en el valor numérico para Costo(i,j).
3. Fases de Ejecución del Pipeline
Para implementar esto de forma modular, dividiremos el código en bloques estrictos:
Pre-procesamiento: Tokenización de oraciones y cálculo de la matriz de embeddings para todo el texto.
Construcción del Grafo (Pruning): Generación de los nodos y aristas válidas utilizando heurísticas de tamaño (ej. mínimo 3 oraciones, máximo 20) y umbrales de embeddings para descartar la basura.
Ponderación de Aristas: Iterar sobre las aristas válidas, consultando al LLM (y guardando en caché) para obtener el coste definitivo de cada segmento candidato.
Búsqueda del Camino Óptimo: Ejecutar el algoritmo de Programación Dinámica (o Dijkstra) sobre el grafo ponderado para recuperar los índices exactos de los cortes.
este es el contexto, no digas nada aun