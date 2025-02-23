import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def moving_average(series, window_size=5):
    """Calculate moving average for a pandas series"""
    return series.rolling(window=window_size, min_periods=1).mean()

def create_game_visualization(game_events, model, model_type="XGBoost", get_probabilities_func=None):
    """
    Create and display game visualization including statistics and animated plot
    
    Parameters:
    game_events (pd.DataFrame): DataFrame containing game events
    model: The model object (XGBoost or Logistic Regression)
    model_type (str): Type of model being visualized
    get_probabilities_func: Function to calculate probabilities for the specific model
    """
    # Calculate probabilities using the provided function
    probabilities = get_probabilities_func(game_events, model)
    game_events['win_probability'] = probabilities
    game_events['win_probability_smooth'] = moving_average(game_events['win_probability'])

    # Display game statistics
    st.header(f"{model_type} Model Game Statistics")
    display_game_stats(game_events)

    # Create and display animated plot
    fig = create_animated_plot(game_events, model_type)
    st.plotly_chart(fig, use_container_width=True)

    # Display events table
    # display_events_table(game_events, model_type)

def display_game_stats(game_events):
    """Display game statistics in a two-column layout"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Win Probability", 
                f"{game_events['win_probability'].iloc[0]:.3f}")
    with col2:
        st.metric("Final Win Probability", 
                f"{game_events['win_probability'].iloc[-1]:.3f}")

    with col3:
        actual_outcome = "Win" if game_events['win'].iloc[0] == 1 else "Loss"
        st.metric("Actual Outcome", actual_outcome)
    with col4:
        st.metric("Final Score Differential", 
                f"{game_events['score_diff'].iloc[-1]}")

def create_animated_plot(game_events, model_type):
    """Create animated plotly figure for win probability with overtime support"""
    fig = go.Figure()

    # Set color based on model type
    if model_type == "XGBoost":
        line_color = "#5D100A" 
    elif model_type == "Logistic Regression":
        line_color = "#154734" 
    else:
        line_color = "blue"

    # Add initial point
    fig.add_trace(
        go.Scatter(
            x=[game_events["time_remaining"].iloc[0]],
            y=[game_events["win_probability_smooth"].iloc[0]],
            mode="lines",
            name="Win Probability",
            line=dict(color=line_color, width=2)
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
                    line=dict(color=line_color, width=2)
                )],
                name=f"frame{i}"
            )
        )
    fig.frames = frames

    # Calculate x-axis range dynamically
    min_time = min(game_events["time_remaining"])
    max_time = 3600

    # Update layout
    fig.update_layout(
        title=f"{model_type} Live Win Probability (Game {game_events['Game_Id'].iloc[0]})",
        xaxis_title="Time Remaining (seconds)",
        yaxis_title="Win Probability",
        xaxis=dict(
            range=[max_time, min_time],  # Dynamic range based on data
            ticktext=['60:00', '45:00', '30:00', '15:00', '0:00', 'OT 5:00', 'OT 10:00', 'OT 15:00', 'OT 20:00'],
            tickvals=[3600, 2700, 1800, 900, 0, -300, -600, -900, -1200],
            tickmode='array'
        ),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        sliders=[{
            "currentvalue": {"prefix": "Time: "},
            "pad": {"t": 50},
            "len": 1,
            "x": 0,
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
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": .5,
            "y": -1.5,
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
        }]
    )
    
    return fig

def display_events_table(game_events, model_type):
    """Display events table with selected columns"""
    st.header(f"{model_type} Game Events")
    display_cols = ['time_remaining', 'score_diff', 'win_probability', 'win_probability_smooth']
    st.dataframe(
        game_events[display_cols].sort_values('time_remaining', ascending=False)
    )