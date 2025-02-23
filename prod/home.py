#######################################################################################################################
# HOMEPAGE - Quick Overview with CTA Buttons and cool background
#######################################################################################################################
import streamlit as st

# Page configuration
st.set_page_config(page_title="GrittyStats", page_icon="🔥", layout="wide")

# Custom CSS for layout
st.markdown("""
    <style>
        .main-content { text-align: center; }
        .cta-button { 
            margin: 10px; 
            width: 220px; /* Ensures all buttons have the same width */
            height: 55px; /* Ensures all buttons have the same height */
            font-size: 18px;
            font-weight: bold;
        }
        .bottom-right { 
            position: fixed; 
            bottom: 20px; 
            right: 20px; 
            width: 100px;
            z-index: -1; /* Moves the image behind all other elements */
        }
        .github-link { text-align: center; margin-top: 50px; }
        .github-icon {
            width: 30px; /* Sets the SVG width */
            height: 30px; /* Sets the SVG height */
            vertical-align: middle;
        }
    </style>
""", unsafe_allow_html=True)

# Center-aligned heading
st.markdown("<h1 class='main-content'>GrittyStats</h1>", unsafe_allow_html=True)

# Call-to-action buttons
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center;'>
            <a href='/page1' target='_self'><button class='cta-button'>View Win Probability Models</button></a>
            <a href='/page2' target='_self'><button class='cta-button'>View Pass Influence Models</button></a>
            <a href='/page3' target='_self'><button class='cta-button'>GrittyLLM</button></a>
            <a href='/page4' target='_self'><button class='cta-button'>Model Evaluation</button></a>
            <a href='/page5' target='_self'><button class='cta-button'>Data Dictionary</button></a>
        </div>
    """, unsafe_allow_html=True)

# Embed an image in the bottom right corner
st.markdown("""
    <img src='https://via.placeholder.com/100' class='bottom-right'>
    """, unsafe_allow_html=True)

# GitHub link with SVG icon
st.markdown("""
     [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/dBCooper2/hacklytics-nhl-dashboard)
    """, unsafe_allow_html=True)
