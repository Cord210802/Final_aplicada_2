import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load the CSV
matches = pd.read_csv("atp_data/atp_matches_till_2022.csv")

# 2) Inspect and clean
# print(matches['minutes'].describe())      # look at distribution & missing
# print(matches['minutes'].isna().sum())  # check for missing values
# nan_percentage = matches['minutes'].isna().mean() * 100
# print(f"Percentage of NaN values in 'minutes': {nan_percentage:.2f}%")  # check for missing values
# total_count = len(matches['minutes'])
# print(f"Total records (including NaN): {total_count}")

df = matches.dropna(subset=['minutes'])   # drop matches without a duration

t_lev = ["G", "M"]

df = df[df['tourney_level'].isin(t_lev)]  # filter for Grand Slam and Masters tournaments

# convert all tournament names to lowercase and strip whitespace
df['tourney_name'] = df['tourney_name'].str.lower().str.strip()

# verify
# sorted(df['tourney_name'].unique())

# if tourney_date is int, first convert to string
df['tourney_date'] = pd.to_datetime(
    df['tourney_date'].astype(str),
    format='%Y%m%d'
)
# now verify
# print(df['tourney_date'].dtype)
# print(df['tourney_date'].head())

df = df.drop(columns=['match_num'])
df = df.drop(columns=["winner_entry", "loser_entry"])  # drop the entry columns
df = df.drop(columns=["winner_ioc", "loser_ioc"]) 

# Variables que causan data leakage (solo conocidas después del partido)
data_leakage_columns = [
    'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_ace', 'w_df', 'w_bpFaced', 'w_bpSaved',
    'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_ace', 'l_df', 'l_bpFaced', 'l_bpSaved',
    'score', 'winner_rank_points', 'loser_rank_points'
]

# Verificar qué columnas existen realmente en el DataFrame
existing_leakage_columns = [col for col in data_leakage_columns if col in df.columns]

# Eliminar las columnas con data leakage
df_clean = df.drop(columns=existing_leakage_columns)
# Mostrar las columnas que quedan disponibles para predecir
# print("\nColumnas disponibles para predicción (sin data leakage):")
remaining_columns = sorted(df_clean.columns.tolist())
# for i, col in enumerate(remaining_columns, 1):
#     print(f"{i:2d}. {col}")

df = df_clean.copy()

# =============================================================================
# PASO 1: SELECCIONAR VARIABLES ALTAMENTE RELEVANTES
# =============================================================================

# Variables altamente relevantes para predicir duración del partido
highly_relevant_vars = [
    'best_of', 'surface', 'round', 'tourney_level',
    'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
    'winner_seed', 'loser_seed', 'minutes'
]

# Crear DataFrame con solo variables relevantes
df_model = df[highly_relevant_vars].copy()

# =============================================================================
# ANÁLISIS DE VALORES FALTANTES
# =============================================================================

# print("=== ANÁLISIS DE VALORES FALTANTES ===\n")

# Contar valores faltantes por columna
missing_counts = df_model.isna().sum()
missing_percentages = (df_model.isna().sum() / len(df_model)) * 100

# Crear DataFrame con la información de valores faltantes
missing_info = pd.DataFrame({
    'Variable': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': missing_percentages.values,
    'Non_Missing_Count': len(df_model) - missing_counts.values
})

# Ordenar por porcentaje de valores faltantes (descendente)
missing_info = missing_info.sort_values('Missing_Percentage', ascending=False)

# print("RESUMEN DE VALORES FALTANTES:")
# print("=" * 60)
# for idx, row in missing_info.iterrows():
#     print(f"{row['Variable']:<15} | Missing: {row['Missing_Count']:>6,} ({row['Missing_Percentage']:>6.2f}%) | Available: {row['Non_Missing_Count']:>6,}")

# print("=" * 60)
# print(f"Total de filas en el dataset: {len(df_model):,}")

# Mostrar estadísticas adicionales
# print(f"\nVARIABLES SIN VALORES FALTANTES:")
no_missing = missing_info[missing_info['Missing_Count'] == 0]['Variable'].tolist()
# if no_missing:
#     for var in no_missing:
#         print(f"  ✓ {var}")
# else:
#     print("  Ninguna variable está completa")

# print(f"\nVARIABLES CON VALORES FALTANTES:")
with_missing = missing_info[missing_info['Missing_Count'] > 0]
# if len(with_missing) > 0:
#     for idx, row in with_missing.iterrows():
#         print(f"  ⚠️  {row['Variable']}: {row['Missing_Percentage']:.2f}% faltante")
# else:
#     print("  Todas las variables están completas")

# Análisis más detallado para cada variable con faltantes
# print("\n=== ANÁLISIS DETALLADO DE VARIABLES CON FALTANTES ===")
for idx, row in with_missing.iterrows():
    var_name = row['Variable']
    if var_name != 'minutes':  # No analizar la variable target aún
        # print(f"\n{var_name.upper()}:")
        # print(f"  - Total faltantes: {row['Missing_Count']:,}")
        # print(f"  - Porcentaje faltante: {row['Missing_Percentage']:.2f}%")
        
        # Mostrar distribución de valores no faltantes
        non_null_values = df_model[var_name].dropna()
        if len(non_null_values) > 0:
            if df_model[var_name].dtype in ['int64', 'float64']:
                pass
                # print(f"  - Estadísticas de valores disponibles:")
                # print(f"    * Media: {non_null_values.mean():.2f}")
                # print(f"    * Mediana: {non_null_values.median():.2f}")
                # print(f"    * Min: {non_null_values.min()}")
                # print(f"    * Max: {non_null_values.max()}")
            else:
                pass
                # print(f"  - Valores únicos disponibles: {non_null_values.nunique()}")
                # print(f"  - Valores más frecuentes:")
                # top_values = non_null_values.value_counts().head(3)
                # for value, count in top_values.items():
                #     pct = (count / len(non_null_values)) * 100
                #     print(f"    * {value}: {count:,} ({pct:.1f}%)")

# Análisis especial para la variable target
if 'minutes' in with_missing['Variable'].values:
    minutes_missing = missing_info[missing_info['Variable'] == 'minutes'].iloc[0]
    # print(f"\nVARIABLE TARGET (minutes):")
    # print(f"  - Faltantes: {minutes_missing['Missing_Count']:,} ({minutes_missing['Missing_Percentage']:.2f}%)")
    # print(f"  - Disponibles: {minutes_missing['Non_Missing_Count']:,}")
    # print(f"  - DECISIÓN: Eliminar filas con minutes faltantes (es nuestra variable target)")
    
df_model = df_model.drop(columns=["loser_seed" , "winner_seed"])  # Eliminar la variable loser_seed

# I want to drop the rows with missing values in the columns winner_rank and loser_rank
df_model = df_model.dropna(subset=['winner_rank', 'loser_rank'])
# Eliminar filas con valores faltantes en la variable target


# =============================================================================
# PASO 2: FEATURE ENGINEERING
# =============================================================================

# print(f"--- FEATURE ENGINEERING ---")

# 2.1 Crear diferencias de ranking y edad
df_model['rank_diff'] = abs(df_model['winner_rank'] - df_model['loser_rank'])
df_model['avg_rank'] = (df_model['winner_rank'] + df_model['loser_rank']) / 2
df_model['min_rank'] = np.minimum(df_model['winner_rank'], df_model['loser_rank'])
# print("✓ Variables de ranking creadas")

df_model['age_diff'] = abs(df_model['winner_age'] - df_model['loser_age'])
df_model['avg_age'] = (df_model['winner_age'] + df_model['loser_age']) / 2
# print("✓ Variables de edad creadas")

# 2.2 Indicadores de paridad
df_model['close_ranking'] = (df_model['rank_diff'] < 50).astype(int)
# print("✓ Variables de paridad creadas")  # check the first few rows of the DataFrame

# Dejar disponible df_model para importación desde otros scripts
cleaned_data = df_model