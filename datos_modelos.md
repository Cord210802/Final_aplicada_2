# Regresion lineal

Starting Tennis Match Duration Analysis...
Dataset shape: (10746, 15)
Target variable (minutes) - Mean: 145.48, Std: 46.28
=== LINEAR REGRESSION MODELS COMPARISON ===


--- Model 1 - Basic ---
Features: 7
Train R²: 0.0477
Test R²: 0.0377
Train RMSE: 45.20
Test RMSE: 45.21
Train MAE: 36.62
Test MAE: 36.59

--- Model 2 - With Player Stats ---
Features: 11
Train R²: 0.0506
Test R²: 0.0419
Train RMSE: 45.13
Test RMSE: 45.11
Train MAE: 36.54
Test MAE: 36.53

--- Model 3 - With Historical Minutes ---
Features: 9
Train R²: 0.0613
Test R²: 0.0518
Train RMSE: 44.88
Test RMSE: 44.88
Train MAE: 36.37
Test MAE: 36.25

--- Model 4 - Full Model ---
Features: 14
Train R²: 0.0644
Test R²: 0.0550
Train RMSE: 44.80
Test RMSE: 44.80
Train MAE: 36.28
Test MAE: 36.19

📊 Model 1 - Basic Summary:
   • Test R²: 0.0377
   • Test RMSE: 45.21 minutes
   • Test MAE: 36.59 minutes
   • Number of features: 7
   • Top 3 most influential features:
     1. surface_Grass: decreases match duration by 9.13 min per unit
     2. surface_Hard: decreases match duration by 2.31 min per unit
     3. round_group_SF: increases match duration by 0.90 min per unit

📊 Model 2 - With Player Stats Summary:
   • Test R²: 0.0419
   • Test RMSE: 45.11 minutes
   • Test MAE: 36.53 minutes
   • Number of features: 11
   • Top 3 most influential features:
     1. surface_Grass: decreases match duration by 9.12 min per unit
     2. surface_Hard: decreases match duration by 2.20 min per unit
     3. round_group_SF: increases match duration by 1.36 min per unit

📊 Model 3 - With Historical Minutes Summary:
   • Test R²: 0.0518
   • Test RMSE: 44.88 minutes
   • Test MAE: 36.25 minutes
   • Number of features: 9
   • Top 3 most influential features:
     1. surface_Grass: decreases match duration by 8.59 min per unit
     2. round_group_SF: increases match duration by 2.17 min per unit
     3. round_group_Other: decreases match duration by 2.14 min per unit

📊 Model 4 - Full Model Summary:
   • Test R²: 0.0550
   • Test RMSE: 44.80 minutes
   • Test MAE: 36.19 minutes
   • Number of features: 14
   • Top 3 most influential features:
     1. surface_Grass: decreases match duration by 8.41 min per unit
     2. round_group_SF: increases match duration by 2.61 min per unit
     3. round_group_Other: decreases match duration by 1.66 min per unit

# XG Boost
Starting Tennis Match Duration XGBoost Analysis...
Dataset shape: (10746, 15)
Target variable (minutes) - Mean: 145.48, Std: 46.28
=== XGBOOST REGRESSION MODELS COMPARISON ===


--- XGB Model 1 - Basic ---
Features: 7
Train R²: 0.2035
Test R²: 0.0121
Train RMSE: 41.34
Test RMSE: 45.80
Train MAE: 33.28
Test MAE: 37.05

--- XGB Model 2 - With Player Stats ---
Features: 11
Train R²: 0.3361
Test R²: 0.0197
Train RMSE: 37.74
Test RMSE: 45.63
Train MAE: 30.14
Test MAE: 36.86

--- XGB Model 3 - With Historical Minutes ---
Features: 9
Train R²: 0.3363
Test R²: 0.0276
Train RMSE: 37.74
Test RMSE: 45.44
Train MAE: 30.22
Test MAE: 36.73

--- XGB Model 4 - Full Model ---
Features: 14
Train R²: 0.3977
Test R²: 0.0289
Train RMSE: 35.95
Test RMSE: 45.41
Train MAE: 28.66
Test MAE: 36.71

XGB Model 1 - Basic Summary:
   • Test R²: 0.0121
   • Test RMSE: 45.80 minutes
   • Test MAE: 37.05 minutes
   • Number of features: 7
   • Top 3 most important features:
     1. loser_rank_points: 0.2539 importance score
     2. winner_rank_points: 0.2147 importance score
     3. surface_Grass: 0.1307 importance score
   ⚠️  High overfitting detected: 0.1914

XGB Model 2 - With Player Stats Summary:
   • Test R²: 0.0197
   • Test RMSE: 45.63 minutes
   • Test MAE: 36.86 minutes
   • Number of features: 11
   • Top 3 most important features:
     1. loser_rank_points: 0.1393 importance score
     2. round_group_Other: 0.1175 importance score
     3. winner_rank_points: 0.1087 importance score
   ⚠️  High overfitting detected: 0.3164

XGB Model 3 - With Historical Minutes Summary:
   • Test R²: 0.0276
   • Test RMSE: 45.44 minutes
   • Test MAE: 36.73 minutes
   • Number of features: 9
   • Top 3 most important features:
     1. loser_rank_points: 0.1568 importance score
     2. winner_rank_points: 0.1287 importance score
     3. round_group_Other: 0.1275 importance score
   ⚠️  High overfitting detected: 0.3087

XGB Model 4 - Full Model Summary:
   • Test R²: 0.0289
   • Test RMSE: 45.41 minutes
   • Test MAE: 36.71 minutes
   • Number of features: 14
   • Top 3 most important features:
     1. loser_rank_points: 0.1203 importance score
     2. w_avg_minutes: 0.0889 importance score
     3. winner_rank_points: 0.0879 importance score
   ⚠️  High overfitting detected: 0.3688

# Random Forest

--- RF Model 1 - Basic ---
Features: 7
Train R²: 0.5718
Test R²: -0.0472
OOB Score: -0.0407
Train RMSE: 30.31
Test RMSE: 47.16
Train MAE: 23.95
Test MAE: 38.07

--- RF Model 2 - With Player Stats ---
Features: 11
Train R²: 0.6718
Test R²: 0.0354
OOB Score: 0.0125
Train RMSE: 26.54
Test RMSE: 45.26
Train MAE: 20.90
Test MAE: 36.53

--- RF Model 3 - With Historical Minutes ---
Features: 9
Train R²: 0.6605
Test R²: 0.0300
OOB Score: 0.0175
Train RMSE: 26.99
Test RMSE: 45.39
Train MAE: 21.25
Test MAE: 36.63

--- RF Model 4 - Full Model ---
Features: 14
Train R²: 0.7135
Test R²: 0.0547
OOB Score: 0.0432
Train RMSE: 24.79
Test RMSE: 44.81
Train MAE: 19.50
Test MAE: 36.21

RF Model 1 - Basic Summary:
   • Test R²: -0.0472
   • OOB R²: -0.0407
   • Test RMSE: 47.16 minutes
   • Test MAE: 38.07 minutes
   • Number of features: 7
   • Top 3 most important features (permutation):
     1. loser_rank_points: 0.0717
     2. winner_rank_points: 0.0138
     3. round_group_Other: 0.0011
   ⚠️  High overfitting detected: 0.6190

RF Model 2 - With Player Stats Summary:
   • Test R²: 0.0354
   • OOB R²: 0.0125
   • Test RMSE: 45.26 minutes
   • Test MAE: 36.53 minutes
   • Number of features: 11
   • Top 3 most important features (permutation):
     1. loser_rank_points: 0.0821
     2. winner_rank_points: 0.0222
     3. w_avg_df: 0.0088
   ⚠️  High overfitting detected: 0.6364


🌲 RF Model 3 - With Historical Minutes Summary:
   • Test R²: 0.0300
   • OOB R²: 0.0175
   • Test RMSE: 45.39 minutes
   • Test MAE: 36.63 minutes
   • Number of features: 9
   • Top 3 most important features (permutation):
     1. loser_rank_points: 0.0688
     2. winner_rank_points: 0.0278
     3. w_avg_minutes: 0.0212
   ⚠️  High overfitting detected: 0.6305

RF Model 4 - Full Model Summary:
   • Test R²: 0.0547
   • OOB R²: 0.0432
   • Test RMSE: 44.81 minutes
   • Test MAE: 36.21 minutes
   • Number of features: 14
   • Top 3 most important features (permutation):
     1. loser_rank_points: 0.0679
     2. winner_rank_points: 0.0198
     3. w_avg_minutes: 0.0147
   ⚠️  High overfitting detected: 0.6588

# GLM Gamma

--- GLM Model 1 - Basic ---
Features: 7
Train R²: 0.0441
Test R²: 0.0289
Pseudo R²: 0.0421
Train RMSE: 45.29
Test RMSE: 45.41
Train MAE: 36.66
Test MAE: 36.70
AIC: 89885.97
BIC: -76894.00

--- GLM Model 2 - With Player Stats ---
Features: 11
Train R²: 0.0475
Test R²: 0.0352
Pseudo R²: 0.0449
Train RMSE: 45.21
Test RMSE: 45.27
Train MAE: 36.57
Test MAE: 36.64
AIC: 89868.88
BIC: -76860.36

--- GLM Model 3 - With Historical Minutes ---
Features: 9
Train R²: 0.0580
Test R²: 0.0423
Pseudo R²: 0.0545
Train RMSE: 44.96
Test RMSE: 45.10
Train MAE: 36.40
Test MAE: 36.38
AIC: 89778.08
BIC: -76887.53

--- GLM Model 4 - Full Model ---
Features: 14
Train R²: 0.0617
Test R²: 0.0472
Pseudo R²: 0.0574
Train RMSE: 44.87
Test RMSE: 44.98
Train MAE: 36.30
Test MAE: 36.30
AIC: 89760.40
BIC: -76845.04

GLM Model 1 - Basic Summary:
   • Test R²: 0.0289
   • Pseudo R² (McFadden): 0.0421
   • Test RMSE: 45.41 minutes
   • Test MAE: 36.70 minutes
   • AIC: 89885.97
   • BIC: -76894.00
   • Number of features: 7
   • Significant coefficients (p < 0.05):
     - loser_rank_points: increases duration (coef=0.0609, p=0.0000)
     - winner_rank_points: decreases duration (coef=-0.0363, p=0.0000)
     - surface_Grass: decreases duration (coef=-0.0630, p=0.0000)
   • Deviance explained: 4.2%
   ✅ Low overfitting: 0.0152

GLM Model 2 - With Player Stats Summary:
   • Test R²: 0.0352
   • Pseudo R² (McFadden): 0.0449
   • Test RMSE: 45.27 minutes
   • Test MAE: 36.64 minutes
   • AIC: 89868.88
   • BIC: -76860.36
   • Number of features: 11
   • Significant coefficients (p < 0.05):
     - loser_rank_points: increases duration (coef=0.0579, p=0.0000)
     - winner_rank_points: decreases duration (coef=-0.0374, p=0.0000)
     - surface_Grass: decreases duration (coef=-0.0636, p=0.0000)
     - l_avg_ace: increases duration (coef=0.0151, p=0.0000)
     - w_avg_ace: decreases duration (coef=-0.0101, p=0.0034)
   • Deviance explained: 4.5%
   ✅ Low overfitting: 0.0123

GLM Model 3 - With Historical Minutes Summary:
   • Test R²: 0.0423
   • Pseudo R² (McFadden): 0.0545
   • Test RMSE: 45.10 minutes
   • Test MAE: 36.38 minutes
   • AIC: 89778.08
   • BIC: -76887.53
   • Number of features: 9
   • Significant coefficients (p < 0.05):
     - loser_rank_points: increases duration (coef=0.0596, p=0.0000)
     - winner_rank_points: decreases duration (coef=-0.0387, p=0.0000)
     - l_avg_minutes: increases duration (coef=0.0254, p=0.0000)
     - w_avg_minutes: increases duration (coef=0.0247, p=0.0000)
     - surface_Grass: decreases duration (coef=-0.0590, p=0.0000)
   • Deviance explained: 5.4%
   ✅ Low overfitting: 0.0157

GLM Model 4 - Full Model Summary:
   • Test R²: 0.0472
   • Pseudo R² (McFadden): 0.0574
   • Test RMSE: 44.98 minutes
   • Test MAE: 36.30 minutes
   • AIC: 89760.40
   • BIC: -76845.04
   • Number of features: 14
   • Significant coefficients (p < 0.05):
     - loser_rank_points: increases duration (coef=0.0569, p=0.0000)
     - winner_rank_points: decreases duration (coef=-0.0410, p=0.0000)
     - w_avg_minutes: increases duration (coef=0.0275, p=0.0000)
     - l_avg_minutes: increases duration (coef=0.0246, p=0.0000)
     - surface_Grass: decreases duration (coef=-0.0584, p=0.0000)
     - w_avg_ace: decreases duration (coef=-0.0140, p=0.0000)
     - l_avg_ace: increases duration (coef=0.0089, p=0.0119)
     - l_avg_df: decreases duration (coef=-0.0083, p=0.0202)
   • Deviance explained: 5.7%
   ✅ Low overfitting: 0.0146

