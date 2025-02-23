import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import dump

# get_prepped_data()
#  - Accesses Data from Google Drive and Preps it for Models
#  - Returns a Test-Train Split as a dictionary with key-value pairs along with the original DataFrame for plotting:
# { 
#   "wp_df" : wp_df
#   "x-test" : X_test,
#   "y-test" : Y_test,
#   "x-train" : X_train,
#   "y-train" : Y_train,
#   "x-train-res" : X_train_res,
#   "y-train-res" : Y_train_res,
# }
def get_prepped_data()->dict:
    # Data Paths :: CHANGE TO GOOGLE DRIVE FOLDER!!!
    pbp_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_pbp20222023.csv'
    shifts_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_shifts20222023.csv'

    # Read in Data ::
    pbp = pd.read_csv(pbp_path)
    shifts = pd.read_csv(shifts_path)

    # Preparing Data ::
    pbp.groupby(['Game_Id', 'Ev_Team'])['Game_Id'].count()

    # Calculate Max Goals for Home and Away Teams for Binary Classifiers for W/L
    pbp['home_max_goal'] = pbp['Home_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['away_max_goal'] = pbp['Away_Score'].groupby(pbp['Game_Id']).transform('max')

    # WP Model DataFrame
    pbp['home_max_goal'] = pbp.groupby('Game_Id')['Home_Score'].transform('max')
    pbp['away_max_goal'] = pbp.groupby('Game_Id')['Away_Score'].transform('max')

    # Calculate Win Observation
    pbp['win'] = np.where(pbp['home_max_goal'] > pbp['away_max_goal'], 1, 0)

    # Fix Time
    pbp['total_seconds_elapsed'] = ((pbp['Period'] - 1) * 1200) + pbp["Seconds_Elapsed"]
    pbp['time_remaining'] = 3600 - pbp['total_seconds_elapsed']

    pbp['score_diff'] = pbp['Home_Score'] - pbp['Away_Score'] # Compute Score Differential

    pbp['skater_diff'] = pbp['Home_Players'] - pbp['Away_Players']
    pbp['goalie_pulled'] = np.where(pbp['Home_Players'] < 5, 1, 0)
    pbp['power_play'] = np.where((pbp['Home_Players'] > pbp['Away_Players']) | (pbp['Away_Players'] > pbp['Home_Players']), 1, 0)

    shifts_agg = shifts.groupby(['Game_Id', 'Player_Id'])['Duration'].sum().reset_index()
    pbp = pbp.merge(shifts_agg, left_on=['Game_Id', 'p1_ID'], right_on=['Game_Id', 'Player_Id'], how='left')
    pbp.rename(columns={'Duration': 'ice_time'}, inplace=True)
    pbp['ice_time'] = pbp['ice_time'].fillna(0)

    wp_df = pbp[['Game_Id', 'p1_name', 'Ev_Team', 'time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time', 'win']]
    wp_df = wp_df.dropna()

    # WP Model :: Test-Train Split
    train_games, test_games = train_test_split(wp_df['Game_Id'].unique(), test_size=0.3, random_state=2142)
    train_df = wp_df[wp_df['Game_Id'].isin(train_games)]
    test_df = wp_df[wp_df['Game_Id'].isin(test_games)]


    X_train = train_df.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1)
    Y_train = train_df[['win']]

    X_test = test_df.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1)
    Y_test = test_df[['win']]


    sm = SMOTE(random_state=2142)
    X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

    res = {
        'pbp_df':pbp,
        'wp_df':wp_df,
        'x-test': X_test,
        'y-test': Y_test,
        'x-train': X_train,
        'y-train': Y_train,
        'x-train-res': X_train_res,
        'y-train-res': Y_train_res,
    }

    return res

def get_xgboost_model(tt_split:dict):
    X_test = tt_split['x-test']
    X_train_res = tt_split['x-train-res']
    Y_train_res = tt_split['y-train-res']
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1)
    xgb_model.fit(X_train_res, Y_train_res.values.ravel())


    Y_pred = xgb_model.predict(X_test)
    Y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    res = {
        'xgb-model':xgb_model,
        'y-pred':Y_pred,
        'y-pred-probs':Y_pred_proba,
    }

    return res

def get_xgb_metrics(tt_split:dict, xgb_model:dict):
    Y_test = tt_split['y-test']
    Y_pred = xgb_model['y-pred']
    Y_pred_proba = xgb_model['y-pred-probs']

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)

    res = {
        'xgb_accuracy': accuracy,
        'xgb_precision': precision,
        'xgb_recall': recall,
        'xgb_roc_auc':roc_auc,
    }

    return res