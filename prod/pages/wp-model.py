import streamlit as st

st.set_page_config(
    page_title="WP Model",  # Tab name in browser
    page_icon="📈",  # Favicon emoji
    layout="centered"  # Ensures better spacing for the buttons
)


col1, col2 = st.columns([.5,1])

with col1:
    st.button('Return Home')


st.title('GrittyStats Win Probability Model')
st.subheader('XGBoost Model for Win Probability')