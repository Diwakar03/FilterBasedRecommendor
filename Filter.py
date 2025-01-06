import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load CSS from an external file
def load_css(css_file):
    with open(css_file, 'r') as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

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

# Load CSS file
load_css("styles.css")

# Streamlit App Title
st.markdown('<h1 class="main-title">Filter-Based Recommendation System</h1>', unsafe_allow_html=True)

# Category Selection
category = st.selectbox(
    "Select a category", options=list(category_files.keys())
)

if category:
    # Load the corresponding files for the selected category
    files = category_files[category]
    vectorizer = load_pickle(files["vectorizer"])
    bow_matrix = load_pickle(files["matrix"])
    dataframe = load_pickle(files["dataframe"])

    # Search bar for user input
    search_query = st.text_input("Search for a name or description")

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
                <div class="recommendation-card">
                    <h4>Name: {row['Name']}</h4>
                    <p>Description: {row['Description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
