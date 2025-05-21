#Cambios buenos

from analisis_grupos import best_of_three_clean, best_of_five_clean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

best_of_five = best_of_five_clean.copy()
best_of_five_clean = best_of_five_clean[best_of_five_clean["tourney_level"] == "G"]


# === Define columns to drop ===
drop_cols = [
    'tourney_date', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc',
    'winner_entry', 'loser_entry','draw_size','tourney_level','match_num','best_of',
    'winner_hand', 'loser_hand','tourney_name', 'round'
    
    # 'minutes' is kept for now as target
]

# === Apply in-place to both datasets ===
best_of_five_clean = best_of_five_clean.drop(columns=drop_cols)
best_of_five_clean = best_of_five_clean.loc[:, ~best_of_five_clean.columns.duplicated()]

# === Define the categorical columns to encode ===
categorical_vars = ['surface']

# === Function: build design matrix with debug and bool-float fix ===
def build_design_matrix(df, categorical_vars):
    for var in categorical_vars:
        if var not in df.columns:
            print(f"❌ Column not found: {var}")
        else:
            print(f"✅ {var} found — dtype: {df[var].dtype}, unique values: {df[var].nunique()}")
    
    # Filter valid categoricals
    valid_cats = [col for col in categorical_vars if col in df.columns]

    print("\n🔄 Applying one-hot encoding...")
    try:
        df_encoded = pd.get_dummies(df, columns=valid_cats, drop_first=True)
        print("✅ One-hot encoding successful.")
    except Exception as e:
        print("❌ Error during get_dummies:")
        print(e)
        return None

    # Convert bools to float (important for correlation)
    for col in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col] = df_encoded[col].astype(float)

    print("\n📐 Shape after encoding:", df_encoded.shape)
    print("🧼 Column types:", df_encoded.dtypes.value_counts())

    # Final numeric design matrix
    numeric_df = df_encoded.select_dtypes(include='number')

    if numeric_df.shape[1] < df_encoded.shape[1]:
        dropped = df_encoded.columns.difference(numeric_df.columns)
        print("\n⚠️ Dropped non-numeric columns:")
        print(dropped.tolist())
    else:
        print("\n✅ All columns are numeric and ready for modeling.")

    return numeric_df

X_design_5 = build_design_matrix(best_of_five_clean, categorical_vars)
corr_5 = X_design_5.corr()


# === Step 1: Define X and y ===
X = X_design_5.drop(columns='minutes')
y = X_design_5['minutes']

# === Step 2: Drop rows with NaNs in specific columns ===
cols_with_nans = [
    'rank_diff', 'avg_rank', 'min_rank',
    'loser_rank', 'loser_rank_points',
    'winner_rank', 'winner_rank_points'
]

# Drop NaNs from X and adjust y accordingly
X_clean = X.dropna(subset=cols_with_nans)
y_clean = y.loc[X_clean.index]

print(f"✅ Rows after dropping NaNs: {X_clean.shape[0]}")

# === Step 3: Remove outliers from y using IQR rule ===
Q1 = y_clean.quantile(0.25)
Q3 = y_clean.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
X_no_outliers = X_clean.loc[mask]
y_no_outliers = y_clean.loc[mask]

print(f"✅ Rows after removing outliers: {X_no_outliers.shape[0]}")