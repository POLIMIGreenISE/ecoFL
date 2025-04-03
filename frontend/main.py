import os
import sys
import streamlit as st

# Add parent directory to Python path to allow imports from other directories
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Configure the page layout and style
st.set_page_config(
    page_title="Eco Federated Learning Platform 🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Simplified navigation with only two main pages
pages = {
    "Federated Learning Simulator 🚀": [
        st.Page("simulator.py", title="FL Simulator"),
    ],
    "Federated Learning Recommender 🔮": [
        st.Page("recommender.py", title="FL Recommender"),
    ]
}

# Create the navigation
pg = st.navigation(pages)
pg.run()