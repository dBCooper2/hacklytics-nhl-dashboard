# Import Packages and Data
import pandas as pd
import numpy as np
import xgboost as xg
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import seaborn as sns
from neo4j import GraphDatabase
import streamlit as st

filename = ''

# Connect to Neo4j
# idk what to put here, but load stuff here
@st.cache_resource
def get_neo4j_session(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver.session()

# Get credentials from Streamlit secrets or user input
NEO4J_URI = st.secrets.database["NEO4J_URI"]
NEO4J_USER = st.secrets.database["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets.database["NEO4J_PASSWORD"]

get_neo4j_session(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Transform Data, Cache in Streamlit

# Once Cleaning Functions are finished for XGBoost, Add Here
@st.cache_resource
def get_pbp_df(fn: str)->pd.DataFrame: # takes in filename, switch to db query?
    return pd.read_csv(fn)

@st.cache_resource
def get_wp_df(pbp:pd.DataFrame)->pd.DataFrame:
    # 1st cleaning cell here...
    # Calculate Max Goals for Home and Away Teams for Binary Classifiers for W/L
    pbp['home_max_goal'] = pbp['Home_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['away_max_goal'] = pbp['Away_Score'].groupby(pbp['Game_Id']).transform('max')

    wp_df = pd.DataFrame()

    wp_df['game_id'] = pbp['Game_Id']
    wp_df['home_team'] = pbp['Home_Team']
    wp_df['home_team'] = pbp['Away_Team']
    wp_df['player1'] = pbp['p1_name'] # does this need to be numeric?
    # Do we need these? The data has it (assuming they are event players)
    wp_df['player2'] = pbp['p2_name']
    wp_df['player3'] = pbp['p3_name']

    wp_df['off_team'] = pbp['Ev_Team']
    wp_df['def_team'] = np.where(pbp['Ev_Team'] != pbp['Home_Team'], pbp['Home_Team'], pbp['Away_Team'])

    # 3600s - time elapsed in seconds, OT games are negative values
    wp_df['time_remaining'] = pbp["Seconds_Elapsed"].apply(lambda x: 3600 - x if x <= 3600 else -(x - 3600))

    pbp['num_home_skaters'] = pbp[['homePlayer1','homePlayer2','homePlayer3','homePlayer4','homePlayer5','homePlayer6']].notna().sum(axis=1)
    pbp['num_away_skaters'] = pbp[['awayPlayer1','awayPlayer2','awayPlayer3','awayPlayer4','awayPlayer5','awayPlayer6']].notna().sum(axis=1)

    wp_df['off_skaters'] = np.where(pbp['Ev_Team'] == pbp['Home_Team'], pbp['num_home_skaters'], pbp['num_away_skaters'])
    wp_df['def_skaters'] = np.where(pbp['Ev_Team'] != pbp['Home_Team'], pbp['num_home_skaters'], pbp['num_away_skaters'])

    wp_df['skater_diff'] = wp_df['off_skaters'] - wp_df['def_skaters']
    wp_df['goalie_pulled'] = np.where(wp_df['off_skaters'] == 6, 1, 0)
    wp_df['off_team_score'] = np.where(pbp['Ev_Team'] == pbp['Home_Team'], pbp['Home_Score'], pbp['Away_Score'])
    wp_df['def_team_score'] = np.where(pbp['Ev_Team'] != pbp['Home_Team'], pbp['Home_Score'], pbp['Away_Score'])
    wp_df['score_diff'] = wp_df['off_team_score'] - wp_df['def_team_score']
    wp_df['xT'] = 0 # ???
    wp_df['off_team_final_score'] = np.where(pbp['Ev_Team'] == pbp['Home_Team'], pbp['home_max_goal'], pbp['away_max_goal'])
    wp_df['def_team_final_score'] = np.where(pbp['Ev_Team'] != pbp['Home_Team'], pbp['home_max_goal'], pbp['away_max_goal'])
    wp_df['win'] = np.where(wp_df['off_team_final_score'] > wp_df['def_team_final_score'],1,0)

    return wp_df

@st.cache_resource
def get_spl_df(wp_df:pd.DataFrame)->pd.DataFrame: # do the splitting stuff here...
    pass

df = get_pbp_df(filename)
wp_df = get_wp_df(df)
spl_df = get_spl_df(wp_df)

# XGBoost Model

def xgboost_functions(df:pd.DataFrame):
    pass

# Streamlit Stuff

def create_wp_viz():
    pass

def create_graph_viz(): # do pyvis for this
    pass

def create_shot_viz(): # hockey-rink with animated WP chart, 100s in-game = 1 irl second
    pass



if __name__ == "__main__":
    pass