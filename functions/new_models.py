import pandas as pd
import io
import boto3
import joblib
import streamlit as st
import numpy as np
import numpy as np
import pandas as pd
from joblib import load
import xgboost as xgb

s3_bucket = 'nhl-data1234'
s3_access_key = st.secrets["AWS"]["AWS_ACCESS_KEY"]
s3_secret_key = st.secrets["AWS"]["AWS_SECRET_KEY"]

def load_parquet_from_s3(object_name):
    s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    print('client created')
    response = s3.get_object(Bucket=s3_bucket, Key=object_name)
    data = response["Body"].read()
    df = pd.read_csv(io.BytesIO(data))
    print('got df')
    return df

def load_pickle_from_s3(object_name):
    """Load a Pickle file from Amazon S3."""
    s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    print('client created')
    response = s3.get_object(Bucket=s3_bucket, Key=object_name)
    data = response["Body"].read()
    model = joblib.load(io.BytesIO(data))
    print('got model')
    return model

@st.cache_resource
def get_prepped_data():
    """Load and process NHL data from S3"""
    pbp = load_parquet_from_s3("nhl_pbp20222023.csv")
    shifts = load_parquet_from_s3("nhl_shifts20222023.csv")

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

def load_parquet_from_s3(object_name):
    s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    print('client created')
    response = s3.get_object(Bucket=s3_bucket, Key=object_name)
    data = response["Body"].read()
    df = pd.read_csv(io.BytesIO(data))
    print('got df')
    return df

def load_pickle_from_s3(object_name):
    """Load a Pickle file from Amazon S3."""
    s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    print('client created')
    response = s3.get_object(Bucket=s3_bucket, Key=object_name)
    data = response["Body"].read()
    model = joblib.load(io.BytesIO(data))
    print('got model')
    return model

@st.cache_resource
def get_prepped_data_rf():
    """Load and process NHL data from S3 FOR RANDOM FOREST"""
    pbp = load_parquet_from_s3("nhl_pbp20222023.csv")
    shifts = load_parquet_from_s3("nhl_shifts20222023.csv")

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
    wp_df = pbp[['Game_Id', 'game_title', 'power_play', 'p1_name', 'Ev_Team', 'time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time', 'win']]
    wp_df = wp_df.dropna()

    return {'wp_df': wp_df}

@st.cache_resource
def get_rf_model():
    """Load the pre-trained Random Forest model from S3."""
    model = load_pickle_from_s3("rf_model.pkl")
    return {'rf-model': model}

@st.cache_resource
def get_lr_model():
    """Load the pre-trained logistic regression model from S3."""
    model = load_pickle_from_s3("logistic_model.pkl")
    return {'lr-model': model}

@st.cache_resource
def get_xgb_model():
    """Load the pre-trained XGBoost model from S3."""
    model = load_pickle_from_s3("xgboost_model.pkl")
    return {'xgb-model': model}

# LR predictions
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

# XGB predictions
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

# RF Prediction
def calculate_rf_probability(game_slice, model):
    """Calculate win probability using the pre-trained Random Forest model"""
    features = game_slice[['time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time']]
    return model.predict_proba(features)[:, 1]

def get_rf_game_probabilities(game_events, model):
    """Calculate win probabilities for all events in a game using Random Forest model"""
    probabilities = []
    for i in range(len(game_events)):
        current_slice = game_events.iloc[i:i+1]
        prob = calculate_rf_probability(current_slice, model)[0]
        probabilities.append(prob)
    return probabilities