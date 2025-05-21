"""
Tennis Five-Set Match Data Analysis & Feature Engineering

This comprehensive script combines data cleaning, feature engineering, 
and analysis for tennis matches, focusing on five-set Grand Slam matches.
It processes tennis match data to prepare it for machine learning model training.

The script performs the following key operations:
1. Load and clean ATP match data
2. Extract only five-set match data from Grand Slam tournaments
3. Create engineered features including ranking differences and player statistics
4. Reshape match data to a player-centric format for historical analysis
5. Calculate rolling averages of player performance metrics
6. Perform multicollinearity analysis using VIF
7. Remove features that would create data leakage
8. Analyze mutual information between categorical features and match duration
9. Generate a final processed dataset ready for modeling

Tennis Statistics Explanation:
- ace: Number of aces
- df: Number of double faults
- svpt: Number of service points
- 1stIn: Number of first serves in
- 1stWon: Number of first serve points won
- 2ndWon: Number of second serve points won
- SvGms: Number of service games
- bpSaved: Number of break points saved
- bpFaced: Number of break points faced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============================================================================================
# STEP 1: DATA LOADING AND CLEANING (from limpieza_grupos.py)
# ============================================================================================

def load_and_clean_data(file_path):
    """
    Load the ATP matches data and perform initial cleaning
    
    Args:
        file_path (str): Path to the ATP matches CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset with initial feature engineering
    """
    print("Loading and cleaning data...")
    
    # 1) Load the CSV
    matches = pd.read_csv(file_path)
    
    # 2) Clean the dataset
    df = matches.dropna(subset=['minutes'])   # drop matches without a duration
    
    # Filter for Grand Slam and Masters tournaments
    t_lev = ["G", "M"]
    df = df[df['tourney_level'].isin(t_lev)]
    
    # Convert all tournament names to lowercase and strip whitespace
    df['tourney_name'] = df['tourney_name'].str.lower().str.strip()
    
    # Convert tourney_date to datetime
    df['tourney_date'] = pd.to_datetime(
        df['tourney_date'].astype(str),
        format='%Y%m%d'
    )
    
    # Create DataFrame for modeling
    df_model = df.copy()
    
    # Drop columns with high missing values
    df_model = df_model.drop(columns=["loser_seed", "winner_seed"])
    
    # Feature Engineering
    print("Creating engineered features...")
    
    # Ranking-based features
    df_model['rank_diff'] = abs(df_model['winner_rank'] - df_model['loser_rank'])
    df_model['avg_rank'] = (df_model['winner_rank'] + df_model['loser_rank']) / 2
    df_model['min_rank'] = np.minimum(df_model['winner_rank'], df_model['loser_rank'])
    
    # Age-based features
    df_model['age_diff'] = abs(df_model['winner_age'] - df_model['loser_age'])
    df_model['avg_age'] = (df_model['winner_age'] + df_model['loser_age']) / 2
    
    # Parity indicators
    df_model['close_ranking'] = (df_model['rank_diff'] < 50).astype(int)
    
    # Surface standardization
    df_model['surface'] = df_model['surface'].replace('Carpet', 'Hard')
    
    print(f"Data cleaning completed. Dataset shape: {df_model.shape}")
    return df_model

# ============================================================================================
# STEP 2: FIVE-SET MATCH PROCESSING (from analisis_grupos.py)
# ============================================================================================

def process_five_set_matches(df):
    """
    Extract and process five-set matches from the dataset
    
    Args:
        df (pd.DataFrame): Cleaned tennis match data
        
    Returns:
        pd.DataFrame: Processed dataset with rolling statistics for five-set matches
    """
    print("Processing five-set matches...")
    
    # Extract only five-set matches
    best_of_five = df[df["best_of"] == 5]
    print(f"Number of five-set matches: {len(best_of_five)}")
    
    # Define the key tennis statistics for feature engineering
    stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']

    # Step 1: Data Reshaping - Convert match data to player-centric format
    
    # Extract winner data and standardize column names
    winner_df = best_of_five[[
        'tourney_date', 'match_num', 'winner_id', 'winner_name'
    ] + [f"w_{s}" for s in stats]].copy()
    winner_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name'] + stats
    winner_df['role'] = 'winner'

    # Extract loser data and standardize column names
    loser_df = best_of_five[[
        'tourney_date', 'match_num', 'loser_id', 'loser_name'
    ] + [f"l_{s}" for s in stats]].copy()
    loser_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name'] + stats
    loser_df['role'] = 'loser'

    # Combine winner and loser data into a unified player-centric dataframe
    long_df = pd.concat([winner_df, loser_df], ignore_index=True)
    
    # Sort by player and chronological order for time-series operations
    long_df = long_df.sort_values(by=['player_id', 'tourney_date', 'match_num'])
    
    # Step 2: Handle Missing Values - Remove rows with missing stats
    long_df = long_df.dropna(subset=stats)

    # Step 3: Feature Engineering - Calculate rolling averages
    
    # Define lookback window for rolling statistics (last 5 matches)
    N = 5  
    
    print(f"Calculating {N}-match rolling averages for player statistics...")
    
    # Calculate rolling average for each statistic
    for s in stats:
        long_df[f"avg_{s}_last_{N}"] = (
            long_df.groupby("player_id")[s]
            .transform(lambda x: x.shift(1).rolling(N, min_periods=1).mean())
        )

    # Step 4: Data Filtering - Ensure sufficient history
    long_df['appearance_order'] = long_df.groupby("player_id").cumcount()
    long_df = long_df[long_df["appearance_order"] >= N - 1].drop(columns=["appearance_order"])

    # Step 5: Re-integration - Merge player historical stats back to match data
    
    # Add winner's historical stats to the match data
    winner_stats = long_df[long_df["role"] == "winner"]
    best_of_five = best_of_five.merge(
        winner_stats[['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats]],
        left_on=['winner_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    best_of_five.rename(columns={f"avg_{s}_last_{N}": f"w_avg_{s}" for s in stats}, inplace=True)
    best_of_five.drop(columns=['player_id'], inplace=True)

    # Add loser's historical stats to the match data
    loser_stats = long_df[long_df["role"] == "loser"]
    best_of_five = best_of_five.merge(
        loser_stats[['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats]],
        left_on=['loser_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    best_of_five.rename(columns={f"avg_{s}_last_{N}": f"l_avg_{s}" for s in stats}, inplace=True)
    best_of_five.drop(columns=['player_id'], inplace=True)

    # Step 6: Remove Incomplete Data
    # List all historical stats columns (both winner and loser)
    agg_cols = [f"w_avg_{s}" for s in stats] + [f"l_avg_{s}" for s in stats]
    
    # Remove matches with missing historical stats
    best_of_five_clean = best_of_five.dropna(subset=agg_cols)
    # Step 7: Remove Data Leakage
    
    # Remove in-match statistics that would cause data leakage
    leakage_cols = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    ]

    # Remove identifiers and post-match information
    id_text_cols = [
        'tourney_id', 
        'winner_id', 'loser_id',
        'score', 'best_of','draw_size'
    ]

    # Combine all columns to drop
    to_drop = leakage_cols + id_text_cols
    
    # Remove these columns from the dataset
    best_of_five_clean = best_of_five_clean.drop(columns=to_drop)

    # Step 8: Final Data Cleaning - Handle duplicate columns
    best_of_five_clean = best_of_five_clean.loc[:, ~best_of_five_clean.columns.duplicated()]
    
    print(f"Five-set match processing completed. Dataset shape: {best_of_five_clean.shape}")
    
    return best_of_five_clean

# ============================================================================================
# STEP 3: GRAND SLAM MATCHES ANALYSIS (from limpieza_grand_slam.py)
# ============================================================================================

def analyze_grand_slam_matches(best_of_five_clean, show_plots=True):
    """
    Filter for Grand Slam matches only and perform further analysis
    
    Args:
        best_of_five_clean (pd.DataFrame): Processed five-set match data
        show_plots (bool): Whether to display analysis plots
        
    Returns:
        pd.DataFrame: Final processed dataset ready for modeling
    """
    print("Analyzing Grand Slam matches...")
    
    # Step 0: Filter Grand Slams only
    best_of_five = best_of_five_clean.copy()
    best_of_five_clean = best_of_five_clean[best_of_five_clean["tourney_level"] == "G"]
    print(f"Number of Grand Slam matches: {len(best_of_five_clean)}")
    
    if show_plots:
        # Step 1: Categorical Analysis for 'tourney_name' and 'round'
        df_check = best_of_five_clean.copy()
        
        for col in ['tourney_name', 'round']:
            if col in df_check.columns:
                plt.figure(figsize=(10, 4))
                sns.countplot(data=df_check, x=col, order=df_check[col].value_counts().index)
                plt.xticks(rotation=45)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.show()
        
        #Distribution of minutes by round
        # Prepare the data: drop any rows missing 'minutes' or 'round'
        df_plot = best_of_five_clean.dropna(subset=['minutes', 'round'])

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_plot,
            x='round',
            y='minutes',
            color='steelblue',    # same color for every box
            showfliers=False      # hide outliers
        )

        plt.title('Distribution of Match Duration (minutes) by Round')
        plt.xlabel('Round')
        plt.ylabel('Minutes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        #Distribution of minutes by surface
        # Prepare the data: drop any rows missing 'minutes' or 'round'
        df_plot = best_of_five_clean.dropna(subset=['minutes', 'surface'])

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_plot,
            x='surface',
            y='minutes',
            color='steelblue',    # same color for every box
            showfliers=False      # hide outliers
        )

        plt.title('Distribution of Match Duration (minutes) by Surface')
        plt.xlabel('Surface')
        plt.ylabel('Minutes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 1) Copy your five‐set DataFrame before dropping leakage cols
        rf_df = best_of_five_clean.copy()

        # 2) Separate target
        y = rf_df['minutes']

        # 3) Build X with only numeric predictors (int64 & float64)
        X = rf_df.drop(columns=['minutes']) \
                .select_dtypes(include=['int64', 'float64'])

        # 4) Drop rows with missing data
        data = pd.concat([X, y], axis=1).dropna()
        X_clean = data[X.columns]
        y_clean = data['minutes']

        # 5) Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )

        # 6) Train the Random Forest regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # 7) Compute feature importances
        importances = pd.Series(rf.feature_importances_, index=X_clean.columns)
        importances = importances.sort_values(ascending=False)

        # 8) Print them
        print("🏆 Random Forest Feature Importances (numerical vars only):")
        for feat, imp in importances.items():
            print(f"{feat:25s} → {imp:.4f}")

        # 9) Plot them
        plt.figure(figsize=(12, 8))
        importances.plot(kind='bar')
        plt.title("RF Feature Importances (Numerical Variables Only)")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    print(best_of_five["tourney_date"].min())
    print(best_of_five["tourney_date"].max())
    # Step 3: Drop irrelevant or leakage-prone columns
    drop_cols = [
        'tourney_date', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc',
        'winner_entry', 'loser_entry', 'tourney_level', 'match_num',
        'winner_hand', 'loser_hand', 'tourney_name', 'round'
    ]
    
    # Only drop columns that actually exist in the dataframe
    drop_cols = [col for col in drop_cols if col in best_of_five_clean.columns]
    
    best_of_five_clean = best_of_five_clean.drop(columns=drop_cols)
    best_of_five_clean = best_of_five_clean.loc[:, ~best_of_five_clean.columns.duplicated()]
    
    # Step 4: Encode surface as dummy
    categorical_vars = ['surface']
    
    def build_design_matrix(df, categorical_vars):
        for var in categorical_vars:
            if var not in df.columns:
                print(f"❌ Column not found: {var}")
            else:
                print(f"✅ {var} found — dtype: {df[var].dtype}, unique values: {df[var].nunique()}")
        
        valid_cats = [col for col in categorical_vars if col in df.columns]
        
        print("\n🔄 Applying one-hot encoding...")
        try:
            df_encoded = pd.get_dummies(df, columns=valid_cats, drop_first=True)
            print("✅ One-hot encoding successful.")
        except Exception as e:
            print("❌ Error during get_dummies:")
            print(e)
            return None
        
        for col in df_encoded.select_dtypes(include='bool').columns:
            df_encoded[col] = df_encoded[col].astype(float)
        
        print("\n📐 Shape after encoding:", df_encoded.shape)
        print("🧼 Column types:", df_encoded.dtypes.value_counts())
        
        numeric_df = df_encoded.select_dtypes(include='number')
        if numeric_df.shape[1] < df_encoded.shape[1]:
            dropped = df_encoded.columns.difference(numeric_df.columns)
            print("\n⚠️ Dropped non-numeric columns:")
            print(dropped.tolist())
        else:
            print("\n✅ All columns are numeric and ready for modeling.")
        
        return numeric_df
    
    X_design_5 = build_design_matrix(best_of_five_clean, categorical_vars)
    
    # Step 5: Define X and y
    X = X_design_5.drop(columns='minutes')
    y = X_design_5['minutes']
    
    # Step 6: Drop rows with NaNs
    cols_with_nans = [
        'rank_diff', 'avg_rank', 'min_rank',
        'loser_rank', 'loser_rank_points',
        'winner_rank', 'winner_rank_points'
    ]
    
    # Only use columns that actually exist in the dataframe
    cols_with_nans = [col for col in cols_with_nans if col in X.columns]
    
    X_clean = X.dropna(subset=cols_with_nans)
    y_clean = y.loc[X_clean.index]
    print(f"✅ Rows after dropping NaNs: {X_clean.shape[0]}")
    
    # Step 7: Remove outliers from y
    Q1 = y_clean.quantile(0.25)
    Q3 = y_clean.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
    X_no_outliers = X_clean.loc[mask]
    y_no_outliers = y_clean.loc[mask]
    print(f"✅ Rows after removing outliers: {X_no_outliers.shape[0]}")
    
    # Step 8: VIF Analysis on numerical-only
    categorical_dummy_cols = [col for col in X_no_outliers.columns if 'surface_' in col]
    numerical_cols = [col for col in X_no_outliers.columns if col not in categorical_dummy_cols]
    
    X_vif_base = X_no_outliers[numerical_cols].copy()
    X_vif = sm.add_constant(X_vif_base)
    
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    print("\n📊 VIF Scores (Numerical Features Only):")
    print(vif_data.sort_values("VIF", ascending=False))
    
    # Step 9: Drop high VIF features (keep categoricals untouched)
    high_vif_features = vif_data[(vif_data['VIF'] > 10) & (vif_data['feature'] != 'const')]['feature'].tolist()
    print(f"\n❌ Dropping high VIF features: {high_vif_features}")
    
    X_final = X_no_outliers.drop(columns=high_vif_features)
    print(f"✅ Final shape after dropping multicollinear numerical features: {X_final.shape}")
    
    # Create final dataset with target variable
    final_data = X_final.copy()
    final_data['minutes'] = y_no_outliers
    
    return final_data

# ============================================================================================
# MAIN EXECUTION
# ============================================================================================

def main(file_path, show_plots=True):
    """
    Execute the complete tennis data analysis pipeline
    
    Args:
        file_path (str): Path to the ATP matches CSV file
        show_plots (bool): Whether to display analysis plots
        
    Returns:
        pd.DataFrame: Final processed dataset ready for modeling
    """
    # Step 1: Load and clean data
    cleaned_data = load_and_clean_data(file_path)
    
    # Step 2: Process five-set matches
    best_of_five_clean = process_five_set_matches(cleaned_data)
    
    # Step 3: Analyze Grand Slam matches
    final_data = analyze_grand_slam_matches(best_of_five_clean, show_plots)
    
    print("\n✨ Tennis data analysis pipeline complete!")
    print(f"Final dataset shape: {final_data.shape}")
    
    return final_data

# Example usage
if __name__ == "__main__":
    file_path = "/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data/atp_matches_till_2022.csv"
    final_data = main(file_path, show_plots=True)
