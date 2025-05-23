# Predicción de la Duración de Partidos de Tenis en Torneos Grand Slam

## 1. Introducción

Nuestro proyecto tiene como objetivo aplicar los modelos lineales vistos en el curso de Estadística Aplicada II a un problema real del ámbito deportivo: la predicción de la duración de partidos de tenis profesional masculino en torneos Grand Slam. A través del análisis estadístico, la limpieza e ingeniería de datos, y la estimación mediante modelos predictivos, se pretende identificar patrones que permitan anticipar la duración de un encuentro utilizando únicamente información disponible antes de su inicio.

## 2. Delimitación y Relevancia del Problema

Para asegurar consistencia y comparabilidad, hemos restringido el universo de estudio a partidos masculinos que se disputaron en los cuatro Grand Slams (Australian Open, Roland Garros, Wimbledon y US Open) entre enero de 1991 y agosto de 2022. Esta decisión elimina la heterogeneidad de formatos (por ejemplo, mejor de tres sets) y aprovecha la riqueza de datos disponible en estas competiciones. Predecir la duración de un partido antes de su inicio permite a organizadores y transmisiones televisivas anticiparse a posibles retrasos o extensiones, y a entrenadores gestionar la carga física de los jugadores.

## 3. Ingeniería de Variables y Prevención de Fuga de Información

Una pieza clave del análisis ha sido evitar la llamada **fuga de información** (data leakage). Para ello, todas las variables predictoras se calcularon sólo con datos anteriores al inicio de cada partido. Concretamente, se construyeron estadísticas agregadas de rendimiento para cada jugador a partir de una **ventana desfasada de cinco encuentros previos**. Estas variables incluyen promedios de aces, dobles faltas, porcentaje de puntos ganados con el primer y segundo servicio, entre otras métricas fundamentales. Al usar únicamente información histórica, garantizamos que el modelo refleja el conocimiento real disponible en el momento de la predicción.

## 4. Transformación y Selección de Variables

Tras cargar y limpiar el conjunto de datos original, eliminamos filas con valores faltantes en variables críticas. Dado que contamos con un volumen muy amplio de partidos (más de 10 000 observaciones), la eliminación de estas filas no provocó una reducción significativa en la información disponible ni en la representatividad del conjunto de entrenamiento. A continuación aplicamos las siguientes transformaciones:

1. **Codificación de variables categóricas**: las variables `surface` (superficie de juego), `round` (ronda del torneo) y `tourney_name` (nombre del torneo) se convirtieron en variables *one-hot* para incluirlas como predictores en los modelos.  
2. **Análisis de colinealidad**: calculamos el Factor de Inflación de Varianza (VIF) para cada variable numérica, descartando aquellas con VIF > 10 para reducir redundancias.  

Como resultado, las variables `round` y `tourney_name` fueron finalmente excluidas del modelo: aunque mostraban correlaciones con la duración, su inclusión elevaba excesivamente la dimensionalidad y presentaba colinealidad alta con otras variables (por ejemplo, superficie), lo cual comprometía la estabilidad y capacidad de generalización de los modelos.

## 5. Modelos Predictivos

### 5.1. Regresión Lineal Múltiple

Primero entrenamos un modelo OLS (Ordinary Least Squares) con las variables numéricas preprocesadas. Este enfoque facilita la interpretación de coeficientes y permite verificar los supuestos clásicos de linealidad, homocedasticidad y normalidad de residuos. Tras ajustar el modelo, generamos los siguientes gráficos de diagnóstico:

- **Residuos vs Valores Ajustados**  
- **Gráfico Q–Q de residuos**  
- **Escala–Local (√|residuo estandarizado| vs ajustados)**  
- **Histograma de residuos con curva normal superpuesta**  

> **Inserta aquí** las cuatro gráficas de diagnóstico resultantes del modelo de regresión lineal, en el mismo orden:  
> 1. Residuos vs Ajustados  
> 2. Q–Q  
> 3. Escala–Local  
> 4. Histograma de residuos  

El análisis de estas gráficas mostró que los supuestos clásicos de la regresión lineal no se cumplen plenamente: en el gráfico **Residuos vs Valores Ajustados** aparece un patrón en forma de embudo (mayor dispersión de residuos a medida que aumentan los valores predichos), lo que evidencia heterocedasticidad; en el **Gráfico Q–Q de Residuos** los puntos extremos se desvían de la línea de referencia, indicando colas más pesadas que las de una distribución normal; el **Histograma de Residuos** presenta un ligero sesgo hacia la derecha y varios valores atípicos; y, aunque en **Leverage vs Residuos** no hay observaciones con leverage excesivo, la **Distancia de Cook** identifica algunos puntos moderadamente influyentes que podrían distorsionar el ajuste global. En conjunto, estos hallazgos sugieren considerar transformaciones de la variable respuesta (por ejemplo, aplicar un logaritmo) o recurrir a métodos más robustos frente a violaciones de homocedasticidad y normalidad (como regresión ponderada o modelos de machine learning).


### 5.2. Random Forest

Para capturar posibles no linealidades e interacciones complejas, complementamos el modelo lineal con un **Random Forest** de 100 árboles. Este método construye múltiples árboles sobre subsamples aleatorias y promedia sus predicciones, lo que suele mejorar la robustez frente al ruido. Empleamos los hiperparámetros:

- `n_estimators = 100`  
- `max_depth = None`  
- `min_samples_split = 2`  
- `min_samples_leaf = 1`  
- `max_features = "auto"`  
- `random_state = 42`  

Posteriormente, sobre el conjunto de test calculamos los residuos (`observado – predicho`) y generamos los mismos cuatro gráficos de diagnóstico.  

> **Inserta aquí** las gráficas de diagnóstico del Random Forest, justo después de la explicación de los hiperparámetros.

### 5.3. XGBoost

Finalmente, probamos un **XGBoost** con 100 iteraciones de boosting secuencial. Este algoritmo optimiza de forma iterativa la función de pérdida (RMSE) y aplica regularización L1/L2 para evitar sobreajuste. Los hiperparámetros más relevantes fueron:

- `n_estimators = 100`  
- `learning_rate = 0.1`  
- `max_depth = 6`  
- `subsample = 1.0`  
- `colsample_bytree = 1.0`  
- `reg_alpha = 0`  
- `reg_lambda = 1`  
- `eval_metric = "rmse"`  
- `use_label_encoder = False`  
- `random_state = 42`  

De igual forma, generamos y colocamos las **cuatro gráficas de diagnóstico** correspondientes al XGBoost.

> **Inserta aquí** las gráficas de XGBoost, después de describir los hiperparámetros.

## 6. Resultados Comparativos e Interpretación

Al comparar los tres enfoques (OLS, Random Forest y XGBoost), observamos lo siguiente:

- La **regresión lineal** ofrece coeficientes claros y supuestos razonablemente cumplidos, pero no captura relaciones no lineales.  
- El **Random Forest** mejora ligeramente el error de predicción (RMSE), pero sin un incremento sustancial, lo que sugiere que con nuestras variables actuales hemos alcanzado un tope de desempeño.  
- El **XGBoost** aporta una ganancia marginal adicional, pero requiere ajuste fino de hiperparámetros y, aun así, muestra riesgo de sobreajuste si no se controla la complejidad de los árboles.

Estos hallazgos indican que, aunque los modelos de árboles permiten explorar interacciones y no linealidades, la clave para avanzar en precisión radica en **incorporar nuevas variables contextuales** (clima, estado físico, historial de enfrentamientos) o en **aplicar técnicas de ingeniería de features** más sofisticadas.

## 7. Conclusiones y Perspectivas Futuras

En resumen, es posible predecir con cierta confiabilidad la duración de un partido de Grand Slam usando únicamente información pre-partido, siempre que se implementen correctamente las etapas de limpieza, selección de variables y prevención de fuga de información. La regresión lineal múltiple resulta un buen punto de partida para entender la influencia de cada predictor, mientras que Random Forest y XGBoost ofrecen flexibilidad adicional para capturar patrones complejos, aunque su beneficio práctico fue limitado en este caso. Para mejorar la precisión en futuras iteraciones, se recomienda:

1. **Explorar variables adicionales** que aporten contexto al desempeño de los jugadores.  
2. **Optimizar hiperparámetros** mediante técnicas de búsqueda sistemática (grid search, Bayesian).  
3. **Investigar otras metodologías** de ensamblado y aprendizaje profundo, siempre cuidando evitar el data leakage.

Con esto, se sientan las bases metodológicas y técnicas para aplicar estadística y machine learning a la optimización de eventos deportivos de alto nivel.
