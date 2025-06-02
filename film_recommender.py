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
        return []

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

# -------------------- SIMPLE COLLABORATIVE FILTERING --------------------
def cf_recommendations(selected_titles, n=10, min_rating=4.0):
    movie_ids = movies[movies['title'].isin(selected_titles)]['movieId'].tolist()
    users_who_liked = ratings[(ratings['movieId'].isin(movie_ids)) & (ratings['rating'] >= min_rating)]['userId'].unique()
    similar_ratings = ratings[(ratings['userId'].isin(users_who_liked)) & (~ratings['movieId'].isin(movie_ids))]
    recommendation_scores = similar_ratings.groupby('movieId')['rating'].mean()
    top_movie_ids = recommendation_scores.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].reset_index(drop=True)

# -------------------- HYBRID MODEL --------------------
def hybrid_recommendations(selected_titles, n=10):
    cbf = content_recommendations(selected_titles, n=30)
    cf = cf_recommendations(selected_titles, n=30)

    cbf_scores = pd.Series([1 - i/30 for i in range(len(cbf))], index=cbf.values)
    cf_scores = pd.Series([1 - i/30 for i in range(len(cf))], index=cf.values)

    hybrid_scores = cbf_scores.add(cf_scores, fill_value=0).sort_values(ascending=False)
    
    return hybrid_scores.head(n).index.tolist()




# -------------------- STREAMLIT UI --------------------
st.title("🎬 Film Öneri Sistemi")
st.write("Beğendiğin filmleri seç, üç farklı sistemden öneri al!")

popular_movies = ratings['movieId'].value_counts().head(300).index
popular_titles = movies[movies['movieId'].isin(popular_movies)]['title'].sort_values().tolist()

selected_movies = st.multiselect("🎥 Beğendiğin filmleri seç:", options=popular_titles)

if st.button("🎯 Önerileri Göster"):
    if selected_movies:
        st.subheader("📚 Content-Based Öneriler:")
        cbf = content_recommendations(selected_movies)
        for title in cbf.index:
            st.write(f"🎬 {title}")

        st.subheader("👥 Collaborative Filtering Öneriler:")
        cf = cf_recommendations(selected_movies)
        for title in cf:
            st.write(f"🎬 {title}")

        st.subheader("🧠 Hybrid Öneriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for title in hybrid:
            st.write(f"🎬 {title}")

    else:
        st.warning("Lütfen en az bir film seç.")
