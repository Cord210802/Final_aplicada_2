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

def load_and_clean_data(file_path, year_start=None, year_end=None):
    """
    Load the ATP matches data, re-label winner/loser by pre-match ranking,
    and perform initial cleaning and feature engineering.
    
    Args:
        file_path (str): Path to the ATP matches CSV file
        year_start (int, optional): Minimum tournament year to include
        year_end   (int, optional): Maximum tournament year to include
        
    Returns:
        pd.DataFrame: Cleaned dataset with original columns plus:
            - winner_/loser_ always refers to the higher/lower ranked player
            - rank_diff, avg_rank, min_rank, age_diff, avg_age, close_ranking
    """
    # 1) Load the CSV
    matches = pd.read_csv(file_path)
    
    # 2) Drop matches without a duration
    df = matches.dropna(subset=['minutes'])
    
    # 3) Filter for Grand Slam and Masters tournaments
    df = df[df['tourney_level'].isin(["G", "M"])]
    
    # 4) Normalize text fields
    df['tourney_name'] = df['tourney_name'].str.lower().str.strip()
    df['tourney_date']  = pd.to_datetime(df['tourney_date'].astype(str),
                                         format='%Y%m%d')
    
    # 5) Optional year filtering
    if year_start is not None:
        df = df[df['tourney_date'].dt.year >= year_start]
    if year_end is not None:
        df = df[df['tourney_date'].dt.year <= year_end]
    
    # 6) Make a working copy and drop high‐missing seed columns
    df_model = df.copy()
    df_model = df_model.drop(columns=["loser_seed", "winner_seed"])
    
    # 7) Re-label winner/loser by pre-match ranking points
    #    Whoever has MORE rank_points becomes 'winner'
    mask = df_model['loser_rank_points'] > df_model['winner_rank_points']
    
    paired_suffixes = [
        # per‐match stats
        '1stIn','1stWon','2ndWon','SvGms','ace','bpFaced','bpSaved','df','svpt',
        # personal / ID fields
        'age','entry','hand','ht','id','ioc','name','rank','rank_points'
    ]
    
    for suf in paired_suffixes:
        wcol = f'w_{suf}'
        lcol = f'l_{suf}'
        if wcol in df_model.columns and lcol in df_model.columns:
            # swap values where mask is True
            df_model.loc[mask, [wcol, lcol]] = df_model.loc[mask, [lcol, wcol]].values
    
    # 8) Recompute derived, non-leaky features
    df_model['rank_diff']     = (df_model['winner_rank'] - df_model['loser_rank']).abs()
    df_model['avg_rank']      = (df_model['winner_rank'] + df_model['loser_rank']) / 2
    df_model['min_rank']      = np.minimum(df_model['winner_rank'], df_model['loser_rank'])
    df_model['age_diff']      = (df_model['winner_age']  - df_model['loser_age']).abs()
    df_model['avg_age']       = (df_model['winner_age'] + df_model['loser_age']) / 2
    df_model['close_ranking'] = (df_model['rank_diff'] < 50).astype(int)
    
    # 9) Standardize surface labels
    df_model['surface'] = df_model['surface'].replace('Carpet', 'Hard')
    
    return df_model

# ============================================================================================
# STEP 2: FIVE-SET MATCH PROCESSING (from analisis_grupos.py)
# ============================================================================================

def process_five_set_matches(df):
    """
    Extract and process five-set matches from the dataset,
    including rolling averages of match stats and match duration.
    
    Args:
        df (pd.DataFrame): Cleaned tennis match data
        
    Returns:
        pd.DataFrame: Processed dataset with rolling statistics for five-set matches
    """
    
    # Extract only five-set matches
    best_of_five = df[df["best_of"] == 5].copy()
    
    # Key tennis statistics for feature engineering
    stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']

    # --- Step 1: Data Reshaping ---
    
    # Winner data
    winner_df = best_of_five[[
        'tourney_date', 'match_num', 'winner_id', 'winner_name', 'minutes'
    ] + [f"w_{s}" for s in stats]].copy()
    winner_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name', 'minutes'] + stats
    winner_df['role'] = 'winner'

    # Loser data
    loser_df = best_of_five[[
        'tourney_date', 'match_num', 'loser_id', 'loser_name', 'minutes'
    ] + [f"l_{s}" for s in stats]].copy()
    loser_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name', 'minutes'] + stats
    loser_df['role'] = 'loser'

    # Combine into long format
    long_df = pd.concat([winner_df, loser_df], ignore_index=True)
    long_df = long_df.sort_values(by=['player_id', 'tourney_date', 'match_num'])
    
    # Drop any rows missing these base stats or minutes
    long_df = long_df.dropna(subset=stats + ['minutes'])

    # --- Step 2: Rolling‐window Feature Engineering ---

    N = 5  # lookback window

    # Rolling averages for each match statistic
    for s in stats:
        long_df[f"avg_{s}_last_{N}"] = (
            long_df
            .groupby("player_id")[s]
            .transform(lambda x: x.shift(1).rolling(N, min_periods=1).mean())
        )

    # Rolling average of match duration (minutes)
    long_df[f"avg_minutes_last_{N}"] = (
        long_df
        .groupby("player_id")['minutes']
        .transform(lambda x: x.shift(1).rolling(N, min_periods=1).mean())
    )

    # Require at least N past appearances
    long_df['appearance_order'] = long_df.groupby("player_id").cumcount()
    long_df = long_df[long_df["appearance_order"] >= N - 1].drop(columns=["appearance_order"])

    # --- Step 3: Merge back into match‐level dataframe ---

    # Winner historical stats
    winner_stats = long_df[long_df["role"] == "winner"]
    merge_cols_w = ['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats] + [f"avg_minutes_last_{N}"]
    best_of_five = best_of_five.merge(
        winner_stats[merge_cols_w],
        left_on=['winner_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    # Rename to w_avg_*
    rename_w = {f"avg_{s}_last_{N}": f"w_avg_{s}" for s in stats}
    rename_w[f"avg_minutes_last_{N}"] = "w_avg_minutes"
    best_of_five.rename(columns=rename_w, inplace=True)
    best_of_five.drop(columns=['player_id'], inplace=True)

    # Loser historical stats
    loser_stats = long_df[long_df["role"] == "loser"]
    merge_cols_l = ['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats] + [f"avg_minutes_last_{N}"]
    best_of_five = best_of_five.merge(
        loser_stats[merge_cols_l],
        left_on=['loser_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    # Rename to l_avg_*
    rename_l = {f"avg_{s}_last_{N}": f"l_avg_{s}" for s in stats}
    rename_l[f"avg_minutes_last_{N}"] = "l_avg_minutes"
    best_of_five.rename(columns=rename_l, inplace=True)
    best_of_five.drop(columns=['player_id'], inplace=True)

    # --- Step 4: Clean up and remove leakage ---

    agg_cols = [f"w_avg_{s}" for s in stats] + ["w_avg_minutes"] + [f"l_avg_{s}" for s in stats] + ["l_avg_minutes"]
    best_of_five_clean = best_of_five.dropna(subset=agg_cols)

    # Drop in‐match stats and identifiers that leak
    leakage_cols = [f"w_{s}" for s in stats] + ["w_minutes"] + [f"l_{s}" for s in stats] + ["l_minutes"]
    id_text_cols = ['tourney_id', 'winner_id', 'loser_id', 'score', 'best_of', 'draw_size']
    to_drop = leakage_cols + id_text_cols
    best_of_five_clean = best_of_five_clean.drop(columns=[c for c in to_drop if c in best_of_five_clean])

    # Remove any duplicated columns
    best_of_five_clean = best_of_five_clean.loc[:, ~best_of_five_clean.columns.duplicated()]

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
    
    # Step 0: Filter Grand Slams only
    best_of_five = best_of_five_clean.copy()
    best_of_five_clean = best_of_five_clean[best_of_five_clean["tourney_level"] == "G"]
    
    if show_plots:
        # Step 1: Categorical Analysis for 'tourney_name' and 'round'
        df_check = best_of_five_clean.copy()
        
        for col in ['tourney_name', 'round']:
            if col in df_check.columns:
                plt.figure(figsize=(10, 4))
                sns.countplot(data=df_check, x=col, order=df_check[col].value_counts().index, color = 'steelblue')
                plt.xticks(rotation=0)
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
        plt.xticks(rotation=0, ha='right')
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
        plt.xticks(rotation=0, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Tendency plot: median match duration by date
        median_minutes_by_date = best_of_five_clean.groupby('tourney_date')['minutes'].median().reset_index()

        plt.figure(figsize=(14, 6))
        plt.plot(median_minutes_by_date['tourney_date'], median_minutes_by_date['minutes'], linestyle='-',color = 'steelblue')
        plt.title('Median Match Duration by Tournament Date (Grand Slams)')
        plt.xlabel('Tournament Date')
        plt.ylabel('Median Match Duration (minutes)')
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
        importances = importances.sort_values()

        # 9) Plot them
        plt.figure(figsize=(12, 8))
        importances.plot(kind='barh',color = 'steelblue')
        plt.title("RF Feature Importances (Numerical Variables Only)")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        #plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

    # Step 3: Drop irrelevant or leakage-prone columns
    drop_cols = [
        'tourney_date', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc',
        'winner_entry', 'loser_entry', 'tourney_level', 'match_num',
        'winner_hand', 'loser_hand', 'tourney_name'
    ]
    
    # Only drop columns that actually exist in the dataframe
    drop_cols = [col for col in drop_cols if col in best_of_five_clean.columns]
    
    best_of_five_clean = best_of_five_clean.drop(columns=drop_cols)
    best_of_five_clean = best_of_five_clean.loc[:, ~best_of_five_clean.columns.duplicated()]
    
    best_of_five_clean['round_group'] = best_of_five_clean['round'].where(
    best_of_five_clean['round'].isin(['F', 'SF', 'QF']),
    other='Other'
    )
    # Step 4: Encode surface as dummy
    categorical_vars = ['surface', 'round_group']

    # 3) Build design matrix with dummies for surface + round_group
    def build_design_matrix(df, categorical_vars):
        valid_cats = [c for c in categorical_vars if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=valid_cats, drop_first=True)
        # convert any bool dummies to float
        for col in df_encoded.select_dtypes(include='bool'):
            df_encoded[col] = df_encoded[col].astype(float)
        # keep only numeric
        return df_encoded.select_dtypes(include='number')

    
    X_design = build_design_matrix(best_of_five_clean, categorical_vars)
    # then proceed as before:
    X = X_design.drop(columns='minutes')
    y = X_design['minutes']
    
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

    
    # Step 7: Remove outliers from y
    Q1 = y_clean.quantile(0.25)
    Q3 = y_clean.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
    X_no_outliers = X_clean.loc[mask]
    y_no_outliers = y_clean.loc[mask]
    
    # Step 8: VIF Analysis on numerical-only
    categorical_dummy_cols = [col for col in X_no_outliers.columns if 'surface_' in col]
    numerical_cols = [col for col in X_no_outliers.columns if col not in categorical_dummy_cols]
    
    X_vif_base = X_no_outliers[numerical_cols].copy()
    X_vif = sm.add_constant(X_vif_base)
    
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    print("\n📊 VIF Scores (Numerical Features Only):")
    print(vif_data.sort_values("VIF", ascending=False).head(3))
    print("...")
    print(vif_data.sort_values("VIF", ascending=False).tail(3))
    
    # Step 9: Drop high VIF features (keep categoricals untouched)
    high_vif_features = vif_data[(vif_data['VIF'] > 10) & (vif_data['feature'] != 'const')]['feature'].tolist()
    print(f"\n❌ Dropping high VIF features")
    
    X_final = X_no_outliers.drop(columns=high_vif_features)
    
    # Create final dataset with target variable
    final_data = X_final.copy()
    final_data['minutes'] = y_no_outliers
    
    return final_data

# ============================================================================================
# MAIN EXECUTION
# ============================================================================================

def main(file_path, show_plots=True, year_start=None, year_end=None):
    """
    Execute the complete tennis data analysis pipeline
    
    Args:
        file_path (str): Path to the ATP matches CSV file
        show_plots (bool): Whether to display analysis plots
        
    Returns:
        pd.DataFrame: Final processed dataset ready for modeling
    """
    # Step 1: Load and clean data
    cleaned_data = load_and_clean_data(file_path, year_start, year_end)
    
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
    final_data = main(file_path, show_plots=False)
    final_data.to_csv("/home/cord2108/ITAM/Aplicada/Proyecto_final/atp_data/final_tennis_data.csv", index=False)