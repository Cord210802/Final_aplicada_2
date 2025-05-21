"""
Tennis Match Data Processing and Feature Engineering Script

This script processes tennis match data to prepare it for machine learning model training. 
It handles both best-of-three and best-of-five match formats, creating historical player
performance features based on rolling averages of key tennis statistics.

The script performs the following key operations:
1. Separates data into best-of-three and best-of-five match datasets
2. Reshapes match data to a player-centric format
3. Calculates rolling averages of player performance metrics
4. Removes features that would create data leakage
5. Cleans and prepares the final datasets for modeling

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

from limpieza_grupos import cleaned_data
import pandas as pd

# Load the data and split by match format
df = cleaned_data
best_of_three = df[df["best_of"] == 3]
best_of_five = df[df["best_of"] == 5]


def process_dataset(dataset):
    """
    Process a tennis dataset to create player performance features based on historical stats.
    
    Args:
        dataset (pd.DataFrame): DataFrame containing tennis match data
        
    Returns:
        pd.DataFrame: Cleaned dataset with engineered features ready for modeling
    """
    # Define the key tennis statistics we'll use for feature engineering
    stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']

    # Step 1: Data Reshaping - Convert match data to player-centric format
    
    # Extract winner data and standardize column names
    winner_df = dataset[[
        'tourney_date', 'match_num', 'winner_id', 'winner_name'
    ] + [f"w_{s}" for s in stats]].copy()
    winner_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name'] + stats
    winner_df['role'] = 'winner'

    # Extract loser data and standardize column names
    loser_df = dataset[[
        'tourney_date', 'match_num', 'loser_id', 'loser_name'
    ] + [f"l_{s}" for s in stats]].copy()
    loser_df.columns = ['tourney_date', 'match_num', 'player_id', 'player_name'] + stats
    loser_df['role'] = 'loser'

    # Combine winner and loser data into a unified player-centric dataframe
    long_df = pd.concat([winner_df, loser_df], ignore_index=True)
    
    # Sort by player and chronological order for time-series operations
    long_df = long_df.sort_values(by=['player_id', 'tourney_date', 'match_num'])
    
    # Step 2: Handle Missing Values - Remove rows with missing stats
    # This is necessary for accurate rolling average calculations
    long_df = long_df.dropna(subset=stats)

    # Step 3: Feature Engineering - Calculate rolling averages
    
    # Define lookback window for rolling statistics (last 5 matches)
    N = 5  
    
    # Calculate rolling average for each statistic
    for s in stats:
        long_df[f"avg_{s}_last_{N}"] = (
            long_df.groupby("player_id")[s]
            .transform(lambda x: x.shift(1).rolling(N, min_periods=1).mean())
        )

    # Step 4: Data Filtering - Ensure sufficient history
    # Remove matches where a player has fewer than N previous appearances
    # This ensures our rolling averages are based on sufficient data
    long_df['appearance_order'] = long_df.groupby("player_id").cumcount()
    long_df = long_df[long_df["appearance_order"] >= N - 1].drop(columns=["appearance_order"])

    # Step 5: Re-integration - Merge player historical stats back to match data
    
    # Add winner's historical stats to the match data
    winner_stats = long_df[long_df["role"] == "winner"]
    dataset = dataset.merge(
        winner_stats[['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats]],
        left_on=['winner_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    dataset.rename(columns={f"avg_{s}_last_{N}": f"w_avg_{s}" for s in stats}, inplace=True)
    dataset.drop(columns=['player_id'], inplace=True)

    # Add loser's historical stats to the match data
    loser_stats = long_df[long_df["role"] == "loser"]
    dataset = dataset.merge(
        loser_stats[['player_id', 'tourney_date', 'match_num'] + [f"avg_{s}_last_{N}" for s in stats]],
        left_on=['loser_id', 'tourney_date', 'match_num'],
        right_on=['player_id', 'tourney_date', 'match_num'],
        how='left'
    )
    dataset.rename(columns={f"avg_{s}_last_{N}": f"l_avg_{s}" for s in stats}, inplace=True)
    dataset.drop(columns=['player_id'], inplace=True)

    # Step 6: Remove Incomplete Data
    # List all historical stats columns (both winner and loser)
    agg_cols = [f"w_avg_{s}" for s in stats] + [f"l_avg_{s}" for s in stats]
    
    # Remove matches with missing historical stats
    dataset_clean = dataset.dropna(subset=agg_cols)

    # Step 7: Remove Data Leakage
    
    # Remove in-match statistics that would cause data leakage
    # These are actual match results that wouldn't be available before the match
    leakage_cols = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    ]

    # Remove identifiers and post-match information
    id_text_cols = [
        'tourney_id', 
        'winner_id', 'loser_id',
        'score'
    ]

    # Combine all columns to drop
    to_drop = leakage_cols + id_text_cols
    
    # Remove these columns from the dataset
    dataset_clean = dataset_clean.drop(columns=to_drop)

    # Step 8: Final Data Cleaning - Handle duplicate columns
    dataset_clean = dataset_clean.loc[:, ~dataset_clean.columns.duplicated()]
    
    return dataset_clean


# Process both datasets
best_of_three_clean = process_dataset(best_of_three)
best_of_five_clean = process_dataset(best_of_five)