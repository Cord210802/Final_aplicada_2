## Análisis de Resultados

En el presente estudio se implementaron cuatro configuraciones progresivas del modelo OLS para evaluar la capacidad predictiva en la duración de partidos de tenis. El Modelo 1 incorpora variables básicas como superficie del torneo, etapa del torneo y puntos de ranking de los jugadores. El Modelo 2 añade estadísticas de desempeño promedio incluyendo aces y dobles faltas. El Modelo 3 integra la duración promedio histórica de partidos por jugador, mientras que el Modelo 4 representa la configuración completa con todas las variables seleccionadas tras el análisis de multicolinealidad. Los resultados revelan que todos los modelos presentan valores de $R^2$ considerablemente bajos, con el modelo más completo explicando únicamente el 5.5% de la varianza en el conjunto de prueba, indicando una capacidad explicativa limitada pero con mejoras progresivas en las métricas de error conforme se incorporan variables adicionales.

### Resultados de Desempeño Predictivo

| Modelo   | R²_train | R²_test | RMSE_train | RMSE_test | MAE_train | MAE_test | N°_Variables |
|----------|----------|---------|------------|-----------|-----------|----------|--------------|
| Modelo 1 | 0.0477   | 0.0377  | 45.20      | 45.21     | 36.62     | 36.59    | 7            |
| Modelo 2 | 0.0506   | 0.0419  | 45.13      | 45.11     | 36.54     | 36.53    | 11           |
| Modelo 3 | 0.0613   | 0.0518  | 44.88      | 44.88     | 36.37     | 36.25    | 9            |
| Modelo 4 | 0.0644   | 0.0550  | 44.80      | 44.80     | 36.28     | 36.19    | 14           |

### Coeficientes Más Relevantes e Interpretación

| Modelo   | Variable          | Coeficiente | Interpretación                                        |
|----------|-------------------|-------------|-------------------------------------------------------|
| Modelo 1 | surface_Grass     | -9.13       | Los partidos en pasto duran 9.13 min menos          |
|          | surface_Hard      | -2.31       | La superficie dura reduce la duración en 2.31 min    |
|          | round_group_SF    | +0.90       | Las semifinales incrementan la duración en 0.90 min  |
| Modelo 2 | round_group_SF    | +1.36       | Aumento mayor al incorporar estadísticas de jugador   |
|          | surface_Grass     | -9.12       | Se mantiene el efecto de superficie                   |
| Modelo 3 | round_group_SF    | +2.17       | Aumenta al incluir minutos históricos                 |
|          | surface_Grass     | -8.59       | Efecto consistente de reducción                       |
| Modelo 4 | surface_Grass     | -8.41       | Superficie sigue siendo el factor más influyente      |
|          | round_group_SF    | +2.61       | Aumenta la duración respecto a rondas anteriores      |
|          | round_group_Other | -1.66       | Rondas iniciales tienen menor duración                |

## Interpretación

El análisis empírico demuestra que la regresión lineal, aun en su configuración más comprehensiva, mantiene una capacidad explicativa limitada coherente con la naturaleza multifactorial inherente a la duración de partidos de tenis, donde influyen variables no observadas como estilos de juego, condiciones climáticas y eventos fortuitos. No obstante, el examen de coeficientes revela patrones sistemáticamente significativos, particularmente el efecto de la superficie de juego, donde los partidos en césped presentan duraciones considerablemente menores, consistente con la dinámica de menor intercambio característica de esta superficie. Las semifinales y finales exhiben incrementos sostenidos en duración, atribuibles a la mayor paridad competitiva en etapas decisivas. Metodológicamente, la regresión lineal demuestra utilidad como modelo base proporcionando transparencia interpretativa y evaluación de relaciones marginales entre variables, estableciendo un benchmark fundamental para la validación de modelos no lineales más complejos, a pesar de su desempeño predictivo comparativamente limitado.