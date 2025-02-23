import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import functions.lr_model as lr
import functions.xgb_model as xgb

# Start overall timer
start_time = time.time()

# Get data and build model
start = time.time()
data_dict = lr.get_prepped_data()
wp_df = data_dict['wp_df']
print(f"Data loading and preparation time: {time.time() - start:.2f} seconds")

# LR Model
start = time.time()
lr_dict = lr.get_lr_model(data_dict)
lr_model = lr_dict['lr-model']
lr_metrics_dict = lr.get_lr_metrics(data_dict, lr_dict)
print(f"Logistic regression model and metrics time: {time.time() - start:.2f} seconds")

# XGB Model
start = time.time()
xgb_dict = xgb.get_xgboost_model(data_dict)
xgb_model = xgb_dict['xgb-model']
xgb_metrics_dict = xgb.get_xgb_metrics(data_dict, xgb_dict)
print(f"XGBoost model and metrics time: {time.time() - start:.2f} seconds")

# Function to calculate win probability
def calculate_win_probability(game_slice, model):
    """Calculate win probability using only data available up to that point"""
    return model.predict_proba(game_slice.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1))[:, 1]

def moving_average(series, window_size=5):
    return series.rolling(window=window_size, min_periods=1).mean()

# Streamlit app layout
st.title("NHL Win Probability Model")

# Display model metrics
st.header("LR Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{lr_metrics_dict['lr_accuracy']:.3f}")
col2.metric("Precision", f"{lr_metrics_dict['lr_precision']:.3f}")
col3.metric("Recall", f"{lr_metrics_dict['lr_recall']:.3f}")
col4.metric("ROC AUC", f"{lr_metrics_dict['lr_roc_auc']:.3f}")

# Game selector
available_games = sorted(wp_df['Game_Id'].unique())
# Game selector for LR model
game_id_lr = st.selectbox("Select a game for LR model:", available_games, key="game_id_lr")


# Get game data and calculate probabilities progressively
start = time.time()
game_events = wp_df[wp_df["Game_Id"] == game_id_lr].sort_values(by="time_remaining", ascending=False)
print(f"Game data selection and sorting time: {time.time() - start:.2f} seconds")

# Calculate win probabilities for each point in time
probabilities = []
start = time.time()
for i in range(len(game_events)):
    # Use only data available up to this point
    current_slice = game_events.iloc[i:i+1]
    prob = calculate_win_probability(current_slice, lr_model)[0]
    probabilities.append(prob)
print(f"Win probability calculation time: {time.time() - start:.2f} seconds")

game_events['win_probability'] = probabilities
game_events['win_probability_smooth'] = moving_average(game_events['win_probability'], window_size=5)

# Create base figure with full data range but empty line
fig = go.Figure()

# Add initial point
fig.add_trace(
    go.Scatter(
        x=[game_events["time_remaining"].iloc[0]],
        y=[game_events["win_probability_smooth"].iloc[0]],
        mode="lines",
        name="Win Probability",
        line=dict(color="blue", width=2)
    )
)

# Create animation frames
frames = []
start = time.time()
for i in range(1, len(game_events)):
    frames.append(
        go.Frame(
            data=[go.Scatter(
                x=game_events["time_remaining"].iloc[:i+1],
                y=game_events["win_probability_smooth"].iloc[:i+1],
                mode="lines",
                line=dict(color="blue", width=2)
            )],
            name=f"frame{i}"
        )
    )
fig.frames = frames
print(f"Animation frames creation time: {time.time() - start:.2f} seconds")

# Update layout with animation controls
start = time.time()
fig.update_layout(
    title=f"LR Live Win Probability (Game {game_id_lr})",
    xaxis_title="Time Remaining (seconds)",
    yaxis_title="Win Probability",
    xaxis=dict(range=[3600, 0]),  # Full game time range
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.1,
        "y": 1.1,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }
                ]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ]
            }
        ]
    }],
    sliders=[{
        "currentvalue": {"prefix": "Time: "},
        "pad": {"t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [f"frame{k}"],
                    {"frame": {"duration": 100, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 0}}
                ],
                "label": str(game_events["time_remaining"].iloc[k]),
                "method": "animate"
            } for k in range(len(frames))
        ]
    }]
)
print(f"Layout update time: {time.time() - start:.2f} seconds")

# Display game statistics
st.header("Game Statistics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Initial Win Probability", 
              f"{game_events['win_probability'].iloc[0]:.3f}")
    st.metric("Final Win Probability", 
              f"{game_events['win_probability'].iloc[-1]:.3f}")

with col2:
    actual_outcome = "Win" if game_events['win'].iloc[0] == 1 else "Loss"
    st.metric("Actual Outcome", actual_outcome)
    st.metric("Final Score Differential", 
              f"{game_events['score_diff'].iloc[-1]}")

# Display the animated chart
st.plotly_chart(fig, use_container_width=True)

# Display key events table
st.header("Game Events")
display_cols = ['time_remaining', 'score_diff', 'win_probability', 'win_probability_smooth']
st.dataframe(
    game_events[display_cols].sort_values('time_remaining', ascending=False)
)

# Display model metrics
st.header("XGB Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{xgb_metrics_dict['xgb_accuracy']:.3f}")
col2.metric("Precision", f"{xgb_metrics_dict['xgb_precision']:.3f}")
col3.metric("Recall", f"{xgb_metrics_dict['xgb_recall']:.3f}")
col4.metric("ROC AUC", f"{xgb_metrics_dict['xgb_roc_auc']:.3f}")

# Game selector
available_games = sorted(wp_df['Game_Id'].unique())
# Game selector for XGB model
game_id_xgb = st.selectbox("Select a game for XGB model:", available_games, key="game_id_xgb")

# XGB Model win probability calculation
game_events_xgb = wp_df[wp_df["Game_Id"] == game_id_xgb].sort_values(by="time_remaining", ascending=False)

probabilities = []
for i in range(len(game_events_xgb)):
    current_slice = game_events_xgb.iloc[i:i+1]
    prob = calculate_win_probability(current_slice, xgb_model)[0]  # Use the XGBoost model
    probabilities.append(prob)

game_events_xgb['win_probability'] = probabilities
game_events_xgb['win_probability_smooth'] = moving_average(game_events_xgb['win_probability'], window_size=5)

# Create base figure with full data range but empty line
fig_xgb = go.Figure()

# Add initial point for XGBoost plot
fig_xgb.add_trace(
    go.Scatter(
        x=[game_events_xgb["time_remaining"].iloc[0]],
        y=[game_events_xgb["win_probability_smooth"].iloc[0]],
        mode="lines",
        name="Win Probability",
        line=dict(color="blue", width=2)
    )
)

# Create animation frames for XGBoost plot
frames_xgb = []
for i in range(1, len(game_events_xgb)):
    frames_xgb.append(
        go.Frame(
            data=[go.Scatter(
                x=game_events_xgb["time_remaining"].iloc[:i+1],
                y=game_events_xgb["win_probability_smooth"].iloc[:i+1],
                mode="lines",
                line=dict(color="blue", width=2)
            )],
            name=f"frame{i}"
        )
    )
fig_xgb.frames = frames_xgb

# Update layout for XGBoost plot
fig_xgb.update_layout(
    title=f"XGB Live Win Probability (Game {game_id_xgb})",
    xaxis_title="Time Remaining (seconds)",
    yaxis_title="Win Probability",
    xaxis=dict(range=[3600, 0]),  # Full game time range
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.1,
        "y": 1.1,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }
                ]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ]
            }
        ]
    }],
    sliders=[{
        "currentvalue": {"prefix": "Time: "},
        "pad": {"t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [f"frame{k}"],
                    {"frame": {"duration": 100, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 0}}
                ],
                "label": str(game_events_xgb["time_remaining"].iloc[k]),
                "method": "animate"
            } for k in range(len(frames_xgb))
        ]
    }]
)

# Display the XGBoost model's plot
st.plotly_chart(fig_xgb, use_container_width=True)

