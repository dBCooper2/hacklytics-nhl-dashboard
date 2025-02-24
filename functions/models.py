import numpy as np
import pandas as pd
from joblib import load
import gdown
import xgboost as xgb
import streamlit as st

import pandas as pd
import requests
import io

# Function to load a pickle file from GitHub LFS
def load_pickle_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses (e.g., 404)
    return pd.read_pickle(io.BytesIO(response.content))



def download_models():
    """Downloads the pre-trained models from Google Drive"""
    # LR Model Download
    # lr_file_id = "11BMLszgKfN1_DFucS6Ihzlz4b6J3IdMT"
    # lr_url = f"https://drive.google.com/uc?id={lr_file_id}"
    # gdown.download(lr_url, "models/logistic_model.pkl", quiet=False)
    
    # XGBoost Model Download
    # xgb_file_id = "1lN_wSnZ6IIr000904ARoDtLuPzAplygE"  # Replace with your XGBoost model file ID
    # xgb_url = f"https://drive.google.com/uc?id={xgb_file_id}"
    # gdown.download(xgb_url, "models/xgboost_model.pkl", quiet=False)

    # Random Forest
    # rf_file_id = "1uwSZFJbJl0wE6gCN1sASikMZ1RnhdRQ4"  # Replace with your XGBoost model file ID
    # rf_url = f"https://drive.google.com/uc?id={rf_file_id}"
    # gdown.download(rf_url, "models/rf_model.pkl", quiet=False)
    lr_url = 'https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_pbp20222023.pkl'
    xgb_url = 'https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_shifts20222023.pkl'

    lr = load_pickle_from_github(lr_url)
    xgb = load_pickle_from_github(xgb_url)


def download_csvs():
    # PBP Data Download
    pbp_file_id = '1rdbu-8ErqmidO5my72wEohsFxrYT7GpZ'  # This is set to public and editor
    # pbp_url = f"https://drive.google.com/uc?id={pbp_file_id}"
    pbp = gdown.download(id=pbp_file_id, output=None, quiet=False)
    
    # Shifts Data Download
    shifts_file_id = '1rtM3Mw8pFc9kZSs2p8XQR_GZfcK6HHkQ'  # Same with this one
    # shifts_url = f"https://drive.google.com/uc?id={shifts_file_id}"
    shifts = gdown.download(id=shifts_file_id, output=None, quiet=False)
    if pbp != None and shifts != None:
        res = {'pbp':pbp, 'shifts':shifts}
        return res
    else:
        print("Neither File was loaded in - TR")

def get_prepped_data()->dict:
    """
    Accesses Data and Preps it for Model Predictions
    Returns a dictionary with processed dataframes
    """
    # Data Paths :: CHANGE TO YOUR DATA PATHS

    try:
        # URLs for raw pickle files (modify if necessary)
        pbp_url = "https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_pbp20222023.pkl"
        shifts_url = "https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_shifts20222023.pkl"

        # Load the pickle files
        pbp = load_pickle_from_github(pbp_url)
        shifts = load_pickle_from_github(shifts_url)

        
    except FileNotFoundError:
        # If files don't exist, download them
        # print("Data files not found locally, downloading from Google Drive...")
        # download_csvs()
        #pbp = pd.read_csv("nhl_pbp20222023.csv")
        #shifts = pd.read_csv("nhl_shifts20222023.csv")
        pbp = pd.read_csv("/mount/src/hacklytics-nhl-dashboard/nhl_pbp20222023.csv")
        shifts = pd.read_csv("/mount/src/hacklytics-nhl-dashboard/nhl_shifts20222023.csv")

    pbp = pbp.replace({
        'Home_Team': {
            'L.A': 'LAK',
            'S.J': 'SJS',
            'N.J': 'NJD',
            'T.B': 'TBL'
        },
        'Away_Team': {
            'L.A': 'LAK',
            'S.J': 'SJS',
            'N.J': 'NJD',
            'T.B': 'TBL'
        }
    })

    # Add Game Title for Streamlit
    nhl_teams = {
        "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
        "BUF": "Buffalo Sabres", "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes",
        "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets",
        "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
        "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
        "MTL": "Montreal Canadiens", "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
        "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
        "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
        "SEA": "Seattle Kraken", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
        "TOR": "Toronto Maple Leafs", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
        "WSH": "Washington Capitals", "WPG": "Winnipeg Jets"
    }

    # Map the tri-codes to full team names
    pbp["Away_Team_Full"] = pbp["Away_Team"].map(nhl_teams)
    pbp["Home_Team_Full"] = pbp["Home_Team"].map(nhl_teams)

    # Create the 'game_title' column using f-strings
    pbp["game_title"] = pbp.apply(lambda row: f"{row['Away_Team_Full']} at {row['Home_Team_Full']} - Game {row['Game_Id']}", axis=1)

    # Preparing Data
    pbp['home_max_goal'] = pbp['Home_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['away_max_goal'] = pbp['Away_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['win'] = np.where(pbp['home_max_goal'] > pbp['away_max_goal'], 1, 0)

    # Fix Time
    pbp['total_seconds_elapsed'] = ((pbp['Period'] - 1) * 1200) + pbp["Seconds_Elapsed"]
    pbp['time_remaining'] = 3600 - pbp['total_seconds_elapsed'].apply(lambda x: 3600 - x if x >= 3600 else -(x-3600))
    
    # Calculate Features
    pbp['score_diff'] = pbp['home_max_goal'] - pbp['away_max_goal']
    pbp['skater_diff'] = pbp['Home_Players'] - pbp['Away_Players']
    pbp['goalie_pulled'] = np.where(pbp['Home_Players'] < 5, 1, 0)
    pbp['power_play'] = np.where((pbp['Home_Players'] > pbp['Away_Players']) | (pbp['Away_Players'] > pbp['Home_Players']), 1, 0)

    # Add ice time
    shifts_agg = shifts.groupby(['Game_Id', 'Player_Id'])['Duration'].sum().reset_index()
    pbp = pbp.merge(shifts_agg, left_on=['Game_Id', 'p1_ID'], right_on=['Game_Id', 'Player_Id'], how='left')
    pbp.rename(columns={'Duration': 'ice_time'}, inplace=True)
    pbp['ice_time'] = pbp['ice_time'].fillna(0)


    # Create final dataframe
    wp_df = pbp[['Game_Id', 'game_title', 'p1_name', 'Ev_Team', 'time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time', 'win']]
    wp_df = wp_df.dropna()

    return {'wp_df': wp_df}

# Logistic Regression Functions
def get_lr_model():
    """Load the pre-trained logistic regression model"""
    try:
        lr_url = 'https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_pbp20222023.pkl'
        model = load_pickle_from_github(lr_url)
        # model = load('logistic_model.pkl')
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
        xgb_url = 'https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/pkl_files/nhl_shifts20222023.pkl'
        model = load_pickle_from_github(xgb_url)
        # model = load('xgboost_model.pkl')
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

def get_rf_model():
    pass

