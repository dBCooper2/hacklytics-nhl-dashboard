import streamlit as st

st.set_page_config(
    page_title="GrittyStats",  # Tab name in browser
    page_icon="🏒",  # Favicon emoji
    layout="centered"  # Ensures better spacing for the buttons
)

st.title('GrittyStats')
st.subheader('NHL Win Probability and Graph Analytics, Powered by Python, AWS s3 and Streamlit')
st.text('Trevor Rowland, Shake Menothu, Anirudh Arunkumar')

col1, col2, col3, col4, col5 = st.columns([.5,1,1,1,.5])
col6, col7, col8, col9 = st.columns([1,1,1,1])

with col1:
    pass
with col2:
    if st.button('GrittyLLM'):
        st.switch_page('pages/gritty-llm.py')
with col3:
    if st.button('Win Probability'):
        st.switch_page('pages/wp-model.py')
with col4:
    if st.button('Graph Analytics'):
        st.switch_page('pages/pass-influence.py')
with col5:
    pass

with col6:
    pass
with col7:
    if st.button('Model Evaluation'):
        st.switch_page('pages/model-eval.py')
with col8:
    if st.button('Data Dictionary'):
        st.switch_page('pages/data-dict.py')
with col9:
    pass

# st.image('https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/site-design/gritty-no-bg.png')
st.image('/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/site-design/gritty-no-bg2.png')



