## Fundamento teórico del modelo y los diagnósticos

### 1. Modelo lineal con interacciones

Partimos del **modelo lineal general**  
$$
\mathbf y = \mathbf X\,\boldsymbol\beta + \boldsymbol\varepsilon,
$$
donde  
- $\mathbf y\in\mathbb R^n$ es el vector de observaciones de la duración de los partidos.  
- $\mathbf X\in\mathbb R^{n\times p}$ es la **matriz de diseño**, que incluye  
  1. Una columna de unos para el intercepto,  
  2. Dummies de superficie (`Grass`, `Hard` versus la referencia `Clay`),  
  3. Variables numéricas base ($\mathbf X_{\text{base}}$),  
  4. Términos de interacción ($\mathbf X_{\text{base}}\times\text{dummies}$).  
- $\boldsymbol\beta\in\mathbb R^p$ contiene  
  - $\beta_0$: intercepto base (superficie de referencia),  
  - $\beta_d$: desplazamientos de intercepto por superficie,  
  - $\gamma_k$: pendientes base de las variables numéricas,  
  - $\delta_{k,d}$: ajustes de pendiente (interacción) para cada superficie.  
- $\boldsymbol\varepsilon$ son los residuos, asumiendo $\mathbb{E}[\varepsilon_i]=0$, $\operatorname{Var}(\varepsilon_i)=\sigma^2$ e independencia.

La estimación por **mínimos cuadrados ordinarios (OLS)** minimiza  
$$
\min_{\boldsymbol b}\,\|\mathbf y - \mathbf X\boldsymbol b\|^2
\quad\longrightarrow\quad
\widehat{\boldsymbol\beta} = (\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y.
$$

### 2. Prevención de colinealidad: VIF

Para garantizar que $\mathbf X^\top\mathbf X$ sea numéricamente estable, calculamos el **Factor de Inflación de Varianza (VIF)** para cada predictor $j$:
$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2},
$$
donde $R_j^2$ es la $R^2$ al regredir la columna $j$ contra el resto de columnas de $\mathbf X$. Valores de VIF > 10 indican colinealidad excesiva y se descartan variables para evitar estimaciones inestables.

### 3. Diagnósticos de ajuste

Una vez ajustado el modelo, usamos los residuos y valores ajustados para comprobar los supuestos:

1. **Residuos vs Ajustados:**  
   Busca patrones sistemáticos. Un embudo o forma curva sugiere **heterocedasticidad**.

2. **Gráfico Q–Q de residuos:**  
   Compara cuantiles empíricos de residuos con cuantiles teóricos normales. Desviaciones en las colas indican **no normalidad**.

3. **Escala–Local** ($\sqrt{|e_i^*|}$ vs predichos):  
   Estima homocedasticidad en escala de residuos estandarizados; la línea LOWESS debe ser aproximadamente horizontal.

4. **Leverage vs Residuos estandarizados:**  
   - **Leverage** ($h_{ii}$) mide la influencia de la $i$-ésima fila en el ajuste  
   - Residuos estandarizados altos ± leverage alto pueden indicar puntos influyentes.

5. **Distancia de Cook:**  
   $$
   D_i = \frac{e_i^2}{p\,\hat\sigma^2}\,\frac{h_{ii}}{(1-h_{ii})^2},
   $$
   cuantifica el cambio en $\widehat{\boldsymbol\beta}$ al eliminar la observación $i$. Valores $D_i>4/n$ señalan observaciones influyentes.

6. **Histograma de residuos con curva normal:**  
   Muestra la distribución de residuos y comprueba asimetría o colas pesadas.

### 4. Interpretación y robustez

- La inclusión de **interacciones** permite que **cada superficie tenga su propia pendiente** para cada variable predictora, capturando efectos contextuales distintos (por ejemplo, cómo `age_diff` impacta diferente en césped versus arcilla).  
- Los diagnósticos confirman o refutan los supuestos de OLS; en presencia de violaciones severas (heterocedasticidad, no normalidad, outliers muy influyentes), podríamos:  
  - Aplicar transformaciones (log–link, Box–Cox).  
  - Usar **regresión ponderada** o modelos de **robustez** (e.g., Huber).  
  - Implementar alternativas **machine learning** más flexibles.

Con esta fundamentación, el bloque de código implementa un análisis exhaustivo que va más allá de la mera estimación de coeficientes, asegurando la validez estadística y la interpretabilidad del modelo.