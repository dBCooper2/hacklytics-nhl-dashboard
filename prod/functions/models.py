import numpy as np
import pandas as pd
from joblib import load
import gdown
import xgboost as xgb

def download_models():
    """Downloads the pre-trained models from Google Drive"""
    # LR Model Download
    lr_file_id = "11BMLszgKfN1_DFucS6Ihzlz4b6J3IdMT"
    lr_url = f"https://drive.google.com/uc?id={lr_file_id}"
    gdown.download(lr_url, "logistic_model.pkl", quiet=False)
    
    # XGBoost Model Download

    xgb_file_id = "1lN_wSnZ6IIr000904ARoDtLuPzAplygE"  # Replace with your XGBoost model file ID
    xgb_url = f"https://drive.google.com/uc?id={xgb_file_id}"
    gdown.download(xgb_url, "xgboost_model.pkl", quiet=False)

def get_prepped_data()->dict:
    """
    Accesses Data and Preps it for Model Predictions
    Returns a dictionary with processed dataframes
    """
    # Data Paths :: CHANGE TO YOUR DATA PATHS
    pbp_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_pbp20222023.csv'
    shifts_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_shifts20222023.csv'

    # Read in Data
    pbp = pd.read_csv(pbp_path)
    shifts = pd.read_csv(shifts_path)

    # Preparing Data
    pbp['home_max_goal'] = pbp['Home_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['away_max_goal'] = pbp['Away_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['win'] = np.where(pbp['home_max_goal'] > pbp['away_max_goal'], 1, 0)

    # Fix Time
    pbp['total_seconds_elapsed'] = ((pbp['Period'] - 1) * 1200) + pbp["Seconds_Elapsed"]
    pbp['time_remaining'] = 3600 - pbp['total_seconds_elapsed']
    
    # Calculate Features
    pbp['score_diff'] = pbp['Home_Score'] - pbp['Away_Score']
    pbp['skater_diff'] = pbp['Home_Players'] - pbp['Away_Players']
    pbp['goalie_pulled'] = np.where(pbp['Home_Players'] < 5, 1, 0)
    pbp['power_play'] = np.where((pbp['Home_Players'] > pbp['Away_Players']) | (pbp['Away_Players'] > pbp['Home_Players']), 1, 0)

    # Add ice time
    shifts_agg = shifts.groupby(['Game_Id', 'Player_Id'])['Duration'].sum().reset_index()
    pbp = pbp.merge(shifts_agg, left_on=['Game_Id', 'p1_ID'], right_on=['Game_Id', 'Player_Id'], how='left')
    pbp.rename(columns={'Duration': 'ice_time'}, inplace=True)
    pbp['ice_time'] = pbp['ice_time'].fillna(0)

    # Create final dataframe
    wp_df = pbp[['Game_Id', 'p1_name', 'Ev_Team', 'time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time', 'win']]
    wp_df = wp_df.dropna()

    return {'wp_df': wp_df}

# Logistic Regression Functions
def get_lr_model():
    """Load the pre-trained logistic regression model"""
    try:
        model = load('logistic_model.pkl')
    except FileNotFoundError:
        print("LR Model not found locally, downloading from Google Drive...")
        download_models()
        model = load('logistic_model.pkl')
    
    return {'lr-model': model}

def calculate_lr_probability(game_slice, model):
    """Calculate win probability using the pre-trained LR model"""
    features = game_slice[['time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time']]
    return model.predict_proba(features)[:, 1]

def get_lr_game_probabilities(game_events, model):
    """Calculate win probabilities for all events in a game using LR model"""
    probabilities = []
    for i in range(len(game_events)):
        current_slice = game_events.iloc[i:i+1]
        prob = calculate_lr_probability(current_slice, model)[0]
        probabilities.append(prob)
    return probabilities

# XGBoost Functions
def get_xgb_model():
    """Load the pre-trained XGBoost model"""
    try:
        model = load('xgboost_model.pkl')
    except FileNotFoundError:
        print("XGBoost Model not found locally, downloading from Google Drive...")
        download_models()
        model = load('xgboost_model.pkl')
    
    return {'xgb-model': model}

def calculate_xgb_probability(game_slice, model):
    """Calculate win probability using the pre-trained XGBoost model"""
    features = game_slice[['time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time']]
    return model.predict_proba(features)[:, 1]

def get_xgb_game_probabilities(game_events, model):
    """Calculate win probabilities for all events in a game using XGBoost model"""
    probabilities = []
    for i in range(len(game_events)):
        current_slice = game_events.iloc[i:i+1]
        prob = calculate_xgb_probability(current_slice, model)[0]
        probabilities.append(prob)
    return probabilities