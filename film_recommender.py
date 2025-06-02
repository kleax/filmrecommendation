import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------- VERI YUKLEME --------------------
@st.cache_data

def load_data():
    movies = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/movies.csv")
    ratings = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/ratings.csv")
    tags = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/tags.csv")

    tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tags_agg, on='movieId', how='left')
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
def cf_recommendations(selected_titles, n=10, min_rating=3.5):
    movie_ids = movies[movies['title'].isin(selected_titles)]['movieId'].tolist()
    users_who_liked = ratings[(ratings['movieId'].isin(movie_ids)) & (ratings['rating'] >= min_rating)]['userId'].unique()
    similar_ratings = ratings[(ratings['userId'].isin(users_who_liked)) & (~ratings['movieId'].isin(movie_ids))]

    recommendation_scores = similar_ratings.groupby('movieId')['rating'].mean()
    top_movie_ids = recommendation_scores.sort_values(ascending=False).head(n).index
    top_movies = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

    return pd.Series(top_movies['title'].values, index=top_movies['title'].values)

# -------------------- HYBRID MODEL --------------------
def hybrid_recommendations(selected_titles, n=10):
    cbf = content_recommendations(selected_titles, n=30)
    cf = cf_recommendations(selected_titles, n=30)

    cbf_scores = pd.Series([1 - i/30 for i in range(len(cbf))], index=cbf.index)
    cf_scores = pd.Series([1 - i/30 for i in range(len(cf))], index=cf.index)

    hybrid_scores = cbf_scores.add(cf_scores, fill_value=0)
    hybrid_scores = hybrid_scores.sort_values(ascending=False).head(n)

    return hybrid_scores.index.tolist()

# -------------------- STREAMLIT UI --------------------
st.title("🎬 Film \u00d6neri Sistemi")
st.write("Be\u011fendi\u011fin filmleri se\u00e7, \u00fc\u00e7 farkl\u0131 sistemden \u00f6neri al!")

popular_movies = ratings['movieId'].value_counts().head(300).index
popular_titles = movies[movies['movieId'].isin(popular_movies)]['title'].sort_values().tolist()

selected_movies = st.multiselect("🎥 Be\u011fendi\u011fin filmleri se\u00e7:", options=popular_titles)

if st.button("🎯 \u00d6nerileri G\u00f6ster"):
    if selected_movies:
        st.subheader("\ud83d\udcc3 Content-Based \u00d6neriler:")
        cbf = content_recommendations(selected_movies)
        for title in cbf.index:
            st.write(f"\ud83c\udfae {title}")

        st.subheader("\ud83d\udc65 Collaborative Filtering \u00d6neriler:")
        cf = cf_recommendations(selected_movies)
        for title in cf.index:
            st.write(f"\ud83c\udfae {title}")

        st.subheader("\ud83e\uddd0 Hybrid \u00d6neriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for title in hybrid:
            st.write(f"\ud83c\udfae {title}")
    else:
        st.warning("L\u00fctfen en az bir film se\u00e7.")
