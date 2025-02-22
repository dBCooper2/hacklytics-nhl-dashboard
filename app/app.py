import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import functions.wpa_model_functions as wpa_mf

# Get data and build model
data_dict = wpa_mf.get_prepped_data()
wp_df = data_dict['wp_df']
lr_dict = wpa_mf.get_lr_model(data_dict)
model = lr_dict['model']
metrics_dict = wpa_mf.get_lr_metrics(data_dict, lr_dict)

def calculate_win_probability(game_slice, model):
    """Calculate win probability using only data available up to that point"""
    return model.predict_proba(game_slice.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1))[:, 1]

def moving_average(series, window_size=5):
    return series.rolling(window=window_size, min_periods=1).mean()

# Streamlit app layout
st.title("NHL Win Probability Model")

# Display model metrics
st.header("Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics_dict['lr_accuracy']:.3f}")
col2.metric("Precision", f"{metrics_dict['lr_precision']:.3f}")
col3.metric("Recall", f"{metrics_dict['lr_recall']:.3f}")
col4.metric("ROC AUC", f"{metrics_dict['lr_roc_auc']:.3f}")

# Game selector
available_games = sorted(wp_df['Game_Id'].unique())
game_id = st.selectbox("Select a game:", available_games)

# Get game data and calculate probabilities progressively
game_events = wp_df[wp_df["Game_Id"] == game_id].sort_values(by="time_remaining", ascending=False)

# Calculate win probabilities for each point in time
probabilities = []
for i in range(len(game_events)):
    # Use only data available up to this point
    current_slice = game_events.iloc[i:i+1]
    prob = calculate_win_probability(current_slice, model)[0]
    probabilities.append(prob)

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

# Update layout with animation controls
fig.update_layout(
    title=f"Live Win Probability (Game {game_id})",
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