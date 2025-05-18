# Variables del Dataset de Tenis - Descripción

| Variable | Descripción | Tipo | Valores |
|----------|-------------|------|---------|
| `tourney_id` | Identificador único del torneo | String | Alfanumérico |
| `tourney_name` | Nombre del torneo | String | Texto |
| `surface` | Superficie en la que se juega el partido | Categórica | Grass, Carpet, Clay, Hard |
| `draw_size` | Tamaño del cuadro del torneo | Numérico | Entero (32, 64, 128, etc.) |
| `tourney_level` | Nivel del torneo | Categórica | G = Grand Slams<br>M = Masters 1000<br>A = Otros eventos ATP Tour<br>C = Challengers<br>S = Satellites/ITFs<br>F = Finales de temporada<br>D = Copa Davis |
| `tourney_date` | Fecha de inicio del torneo | Fecha | YYYYMMDD |
| `match_num` | Número de partido en un torneo específico | Numérico | Entero |
| `id` | Identificador del jugador | String | Alfanumérico |
| `seed` | Cabeza de serie del jugador en el torneo | Numérico | Entero o NaN |
| `entry` | Forma de entrada al torneo | Categórica | WC = Wildcard<br>Q = Clasificado<br>LL = Lucky loser<br>PR = Protected ranking<br>SE = Special Exempt<br>ALT = Alternate player |
| `name` | Nombre del jugador | String | Texto |
| `hand` | Mano dominante del jugador | Categórica | R = Diestro<br>L = Zurdo |
| `ht` | Altura del jugador | Numérico | Entero (en cm) |
| `IOC` | País de origen (código olímpico) | Categórica | Códigos de 3 letras |
| `age` | Edad del jugador | Numérico | Decimal |
| `score` | Resultado final del partido | String | Texto formateado |
| `best_of` | Número máximo de sets en el partido | Numérico | 3 o 5 |
| `round` | Ronda del torneo | Categórica | R128 = Ronda de 128<br>R64 = Ronda de 64<br>R32 = Ronda de 32<br>R16 = Octavos de final<br>QF = Cuartos de final<br>SF = Semifinales<br>F = Final<br>RR = Round Robin<br>ER = Early Round<br>BR = Bronze medal match |
| `minutes` | Duración del partido en minutos | Numérico | Entero |
| `ace` | Número de aces en el partido | Numérico | Entero |
| `df` | Dobles faltas | Numérico | Entero |
| `svpt` | Puntos de servicio | Numérico | Entero |
| `1stin` | Primer servicio dentro (cantidad) | Numérico | Entero |
| `1stWon` | Puntos ganados con primer servicio | Numérico | Entero |
| `2ndWon` | Puntos ganados con segundo servicio | Numérico | Entero |
| `SvGms` | Número de juegos al servicio | Numérico | Entero |
| `bpSaved` | Puntos de break salvados | Numérico | Entero |
| `bpFaced` | Puntos de break enfrentados | Numérico | Entero |

## Notas Adicionales:

- Para las estadísticas de partido (ace, df, svpt, etc.), hay dos versiones de cada variable:
  - Prefijo `w_` para el ganador del partido (winner)
  - Prefijo `l_` para el perdedor del partido (loser)