import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------- VERI YUKLEME --------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")

    tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tags_agg, on='movieId', how='left')

    # genres alanını düzelt
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

    # içerik sütununu zenginleştir
    movies['content'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['tag'].fillna('')

    return movies, ratings

movies, ratings = load_data()

# -------------------- CONTENT-BASED MODEL --------------------
@st.cache_resource
def build_cbf_model(movies):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_cbf_model(movies)

def content_recommendations(selected_titles, n=10):
    valid_titles = [t for t in selected_titles if t in indices]
    if not valid_titles:
        return pd.Series([])

    all_scores = pd.Series(dtype=float)
    for title in valid_titles:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+20]
        for i, score in sim_scores:
            movie_title = movies['title'].iloc[i]
            if movie_title not in selected_titles:
                all_scores[movie_title] = all_scores.get(movie_title, 0) + score

    return pd.Series(all_scores).sort_values(ascending=False).head(n)

# -------------------- COLLABORATIVE FILTERING --------------------
def cf_recommendations(selected_titles, n=10, min_rating=3._
