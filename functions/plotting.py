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

    fig.update_layout(
        width=800,  # Adjust width to your preference
        height=600   # Adjust height to your preference
    )
    
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
    """Create animated plotly figure for win probability with fixed overtime support"""
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

    # Calculate the range dynamically based on the game events
    min_time = min(game_events["time_remaining"])
    max_time = 3600  # Start at 3600 seconds (start of the game)
    
    # Adjust min_time to the nearest lower multiple of 300 (5 minutes) for clean OT periods
    if min_time < 0:
        min_time = -300 * (abs(min_time) // 300 + 1)

    # Create dynamic tick values and labels
    base_ticks = [3600, 2700, 1800, 900, 0, -900]  # Regular time ticks
    base_labels = ['75:00','60:00', '45:00', '30:00', '15:00', '0:00']
    
    # Add OT ticks if needed
    if min_time < 0:
        ot_period = 1
        current_time = -300
        while current_time >= min_time:
            base_ticks.append(current_time)
            minutes = abs(current_time) // 60
            base_labels.append(f'OT{ot_period} {minutes}:00')
            current_time -= 300
            if current_time % 1200 == 0:  # Every 20 minutes, new OT period
                ot_period += 1

    # Update layout with dynamic range and labels
    fig.update_layout(
        title=f"{model_type} Live Win Probability (Game {game_events['Game_Id'].iloc[0]})",
        xaxis_title="Time Remaining (MM:SS)",
        yaxis_title="Win Probability",
        xaxis=dict(
            range=[max_time, min_time],  # Dynamic range based on game events
            ticktext=base_labels,
            tickvals=base_ticks,
            tickmode='array',
            title_standoff=10
        ),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        # Add grid lines
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
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
                    "label": f"{int(game_events['time_remaining'].iloc[k])}",
                    "method": "animate"
                } for k in range(len(frames))
            ]
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": .5,
            "y": -.3,
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