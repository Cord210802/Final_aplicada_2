# Introducción al Problema: Predicción de la Duración de Partidos de Tenis

## Contexto del Problema

En el mundo del tenis profesional, la duración de un partido es una variable crucial que impacta múltiples aspectos de este deporte. Desde la planificación de transmisiones televisivas hasta la preparación física de los jugadores, conocer con anticipación cuánto tiempo durará un encuentro representa un valor significativo para diferentes stakeholders.

Los partidos de tenis pueden variar dramáticamente en duración: desde encuentros que se resuelven en menos de una hora hasta épicas batallas que se extienden por más de cinco horas. Esta variabilidad está influenciada por múltiples factores como el formato del torneo (mejor de 3 o 5 sets), la superficie de juego, las características de los jugadores involucrados, y la etapa del torneo en la que se desarrolla el partido.

## Relevancia del Problema

La predicción precisa de la duración de partidos de tenis tiene implicaciones prácticas importantes:

### 1. **Gestión de Transmisiones y Media**
- Las cadenas televisivas necesitan planificar su programación con precisión
- Los servicios de streaming requieren estimaciones para optimizar la asignación de recursos
- La planificación publicitaria se beneficia de predicciones más exactas

### 2. **Operaciones de Torneos**
- Programación eficiente de partidos en las diferentes canchas
- Gestión de personal (árbitros, recogepelotas, personal médico)
- Planificación logística para espectadores (transporte, catering, etc.)

### 3. **Preparación de Jugadores y Entrenadores**
- Estrategias de preparación física específicas
- Planificación nutricional e hidratación
- Gestión de energía durante torneos largos

## Preguntas de Investigación

Este análisis estadístico busca responder las siguientes preguntas de investigación:

### Pregunta Principal:
**¿Es posible predecir la duración de un partido de tenis utilizando únicamente información disponible antes de que comience el encuentro?**

### Preguntas Específicas:

1. **¿Qué variables pre-partido tienen mayor influencia en la duración del encuentro?**
   - ¿Cómo afecta el formato del partido (mejor de 3 vs. 5 sets)?
   - ¿Cuál es el impacto de la superficie de juego (arcilla, césped, pista dura)?
   - ¿La etapa del torneo influye en la duración esperada?

2. **¿Las características de los jugadores son predictores significativos?**
   - ¿La diferencia de ranking entre jugadores afecta la duración?
   - ¿La edad de los competidores tiene relevancia predictiva?

3. **¿Existen patrones específicos según el nivel del torneo?**
   - ¿Los Grand Slams tienen comportamientos diferentes a otros torneos?
   - ¿Cómo varían las duraciones entre Masters 1000?

4. **¿Qué nivel de precisión es alcanzable en estas predicciones?**
   - ¿Cuál es el error promedio esperado en las predicciones?
   - ¿Existen ciertos tipos de partidos más predecibles que otros?

## Enfoque Metodológico

Para abordar estas preguntas, utilizaremos técnicas de modelado estadístico, específicamente regresión lineal, aplicadas a un dataset histórico de partidos de tenis profesional masculino. El análisis se centrará en variables disponibles antes del inicio del partido, evitando cualquier filtración de información que comprometería la aplicabilidad práctica del modelo.

Este enfoque nos permitirá no solo desarrollar un modelo predictivo funcional, sino también generar insights valiosos sobre los factores que determinan la duración de los encuentros tenísticos en el circuito profesional.

---
# Justificación del Enfoque Estadístico

## Delimitación del Universo de Análisis

El presente análisis se enfoca exclusivamente en partidos correspondientes a torneos del **circuito Grand Slam** masculino, caracterizados por jugarse al **mejor de cinco sets**. Esta restricción no es arbitraria: responde a una decisión metodológica que busca **reducir la heterogeneidad estructural del problema** y mejorar la precisión de las inferencias y predicciones. En comparación con otros torneos del circuito ATP (donde los partidos se juegan al mejor de tres sets), los Grand Slams exhiben:

- una duración promedio significativamente mayor,
- una mayor varianza en el tiempo de juego, y
- dinámicas particulares asociadas a su estructura competitiva.

Esta homogeneidad relativa en las reglas permite asumir mayor estabilidad en la relación entre las variables predictoras y la variable objetivo, que en este caso es la **duración del partido en minutos**.

## Criterios para la Selección de Variables

Uno de los pilares del análisis fue la construcción cuidadosa del conjunto de variables explicativas. El objetivo fue identificar únicamente aquellas variables que:

1. Están **disponibles antes del comienzo del partido** (excluyendo cualquier variable que implique información sobre el resultado o evolución del encuentro).
2. Son estadísticamente informativas respecto a la duración del partido.
3. No introducen problemas de colinealidad severa ni redundancia estructural.

El proceso de selección se llevó a cabo en múltiples etapas:

### 1. Evaluación de Variables Categóricas: `tourney_name` y `round`

Estas dos variables fueron objeto de análisis específico mediante:

- **Visualización de distribución de frecuencias**, para verificar la representatividad de sus categorías.
- **Información mutua (mutual information)** respecto a la variable `minutes`, con el fin de cuantificar dependencia sin asumir linealidad.

Ambas demostraron contener información relevante: la variable `round`, por ejemplo, está asociada a un patrón de incremento en la duración media a medida que se avanza en el torneo, mientras que `tourney_name` captura diferencias estructurales entre superficies y condiciones particulares de cada Slam.

### 2. Preprocesamiento y codificación

Las variables categóricas (`surface`, `tourney_name`, `round`) fueron transformadas mediante **codificación one-hot**, manteniendo todas sus componentes tras eliminar la primera categoría (dummy encoding con `drop_first=True`) para evitar dependencia lineal exacta. Se optó por preservar estas variables codificadas en su totalidad, independientemente de su correlación con otras, debido a su **valor explicativo intrínseco no colineal** y su **interpretabilidad contextual**.

### 3. Diagnóstico de multicolinealidad en variables numéricas

Para el conjunto de variables numéricas, se llevó a cabo un **análisis de colinealidad mediante el factor de inflación de varianza (VIF)**. Las variables con VIF significativamente superiores a 10 fueron eliminadas del diseño del modelo, al considerarse problemáticas para la estimación estable de coeficientes. Este paso se restringió a variables numéricas; las variables categóricas codificadas fueron excluidas de este filtro para preservar su rol estructural en el modelo.

## Prevención del Fuga de Información (*Data Leakage*)

Un aspecto crítico en todo problema de predicción es evitar el **uso de información futura en variables supuestamente predictoras**, lo que llevaría a un *overfitting* artificial y resultados irreproducibles en escenarios reales.

En este análisis, abordamos explícitamente este riesgo al construir **agregados estadísticos de los jugadores (por ejemplo, porcentajes de primer servicio ganado, aces, dobles faltas, etc.) mediante ventanas temporales desfasadas**. Específicamente:

- Para cada partido, las métricas del jugador se calcularon **únicamente usando partidos previos a esa fecha**, ordenados cronológicamente.
- Se utilizaron ventanas móviles con tamaño controlado para asegurar que las estadísticas sean representativas pero nunca contaminadas por eventos futuros.
- Esta estrategia garantiza que las variables construidas reflejan conocimiento históricamente observable **en el momento justo antes del partido**, simulando las condiciones reales de toma de decisión o pronóstico.

Esta precaución metodológica es clave para asegurar la validez del modelo y su aplicabilidad en entornos productivos o simulaciones prospectivas.

