import streamlit as st
import time
import pandas as pd
import functions.models as wp
from functions.plotting import create_game_visualization

# Page config
st.set_page_config(page_title="NHL Win Probability", layout="wide")

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load data and models
@st.cache_resource
def load_models_and_data():
    data_dict = wp.get_prepped_data()
    lr_dict = wp.get_lr_model()
    xgb_dict = wp.get_xgb_model()
    return data_dict, lr_dict, xgb_dict

# App title
st.title("NHL Win Probability Models")

try:
    # Load data and models with a loading spinner
    with st.spinner('Loading data and models...'):
        data_dict, lr_dict, xgb_dict = load_models_and_data()
        wp_df = data_dict['wp_df']
        lr_model = lr_dict['lr-model']
        xgb_model = xgb_dict['xgb-model']
        st.session_state.model_loaded = True

    # Game selector
    available_games = sorted(wp_df['game_title'].unique())
    game_id = st.selectbox("Select a game:", available_games)

    # Get game data
    game_events = wp_df[wp_df["game_title"] == game_id].sort_values(by="time_remaining", ascending=False)

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["XGBoost Model", "Logistic Regression Model"])

    with tab1:
        create_game_visualization(
            game_events.copy(), 
            xgb_model, 
            "XGBoost",
            wp.get_xgb_game_probabilities
        )

    with tab2:
        create_game_visualization(
            game_events.copy(), 
            lr_model, 
            "Logistic Regression",
            wp.get_lr_game_probabilities
        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data paths and model availability.")