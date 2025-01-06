import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved files
def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Dictionary to map category options to their corresponding files
category_files = {
    "Anime": {
        "vectorizer": "Fdf1_bow_vectorizer.pkl",
        "matrix": "Fdf1_bow_matrix.pkl",
        "dataframe": "Fdf1.pkl",
    },
    "Hindi Show": {
        "vectorizer": "Fdf2_bow_vectorizer.pkl",
        "matrix": "Fdf2_bow_matrix.pkl",
        "dataframe": "Fdf2.pkl",
    },
    "Hindi Movies": {
        "vectorizer": "Fdf3_bow_vectorizer.pkl",
        "matrix": "Fdf3_bow_matrix.pkl",
        "dataframe": "Fdf3.pkl",
    },
    "English Movies": {
        "vectorizer": "Fdf4_bow_vectorizer.pkl",
        "matrix": "Fdf4_bow_matrix.pkl",
        "dataframe": "Fdf4.pkl",
    },
    "English TV Shows": {
        "vectorizer": "Fdf5_bow_vectorizer.pkl",
        "matrix": "Fdf5_bow_matrix.pkl",
        "dataframe": "Fdf5.pkl",
    },
}

# Streamlit App Title
st.markdown('<h1 style="color:white; text-align:left; font-size:36px; margin:20px 20px; font-weight:bold;">Filter-Based Recommendation System</h1>', unsafe_allow_html=True)

# Category Selection with Inline CSS
category = st.selectbox(
    "Select a category", options=list(category_files.keys()), 
    index=0, key="category", 
    help="Select a category", 
    label_visibility="visible"
)
st.markdown('<style>div.stSelectbox > label { color: black !important; font-size: 16px !important; font-weight: bold !important; }</style>', unsafe_allow_html=True)

# Search bar for user input with Inline CSS
search_query = st.text_input("Search for a name or description")
st.markdown('<style>div.stTextInput > label { color: black !important; font-size: 16px !important; font-weight: bold !important; }</style>', unsafe_allow_html=True)

if category:
    # Load the corresponding files for the selected category
    files = category_files[category]
    vectorizer = load_pickle(files["vectorizer"])
    bow_matrix = load_pickle(files["matrix"])
    dataframe = load_pickle(files["dataframe"])

    if search_query:
        # Preprocess the search query
        search_vector = vectorizer.transform([search_query])
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(search_vector, bow_matrix).flatten()
        # Get the top 7 recommendations
        top_indices = similarity_scores.argsort()[-7:][::-1]
        recommendations = dataframe.iloc[top_indices]

        # Display Recommendations
        st.markdown(f"<h3 style='color:white;'>Top 7 recommendations for '{search_query}':</h3>", unsafe_allow_html=True)
        for i, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="background-color:white; border:2px solid #87ceeb; border-radius:8px; padding:15px; margin-bottom:15px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                    <h4 style="margin:0; font-size:18px; font-weight:bold; color:#333333;">Name: {row['Name']}</h4>
                    <p style="margin:5px 0 0; font-size:14px; color:#555555;">Description: {row['Description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
