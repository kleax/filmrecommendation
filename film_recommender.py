import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------- VERİ YÜKLEME --------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")

    tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tags_agg, on='movieId', how='left')
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
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

    valid_movies = ratings['movieId'].value_counts()
    valid_movies = valid_movies[valid_movies > 10].index
    similar_ratings = similar_ratings[similar_ratings['movieId'].isin(valid_movies)]

    recommendation_scores = similar_ratings.groupby('movieId')['rating'].mean()
    top_movie_ids = recommendation_scores.sort_values(ascending=False).head(n).index
    top_movies = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

    return pd.Series(top_movies['title'].values, index=top_movies['title'].values)

# -------------------- HYBRID MODEL --------------------
def hybrid_recommendations(selected_titles, n=10):
    cbf = content_recommendations(selected_titles, n=30)
    cf = cf_recommendations(selected_titles, n=30)

    cbf_scores = pd.Series([1 - i / 30 for i in range(len(cbf))], index=cbf.index)
    cf_scores = pd.Series([1 - i / 30 for i in range(len(cf))], index=cf.index)

    hybrid_scores = cbf_scores.add(cf_scores, fill_value=0)
    hybrid_scores = hybrid_scores.sort_values(ascending=False).head(n)

    return hybrid_scores.index.tolist()

# -------------------- STREAMLIT UI --------------------
st.title("🎬 Film Öneri Sistemi")
st.markdown("Beğendiğin filmleri seç, sistem senin için öneri yapsın.")
st.markdown(f"📊 Sistem, **{len(movies):,} film** ve **{len(ratings):,} kullanıcı oyu** ile çalışıyor.")

popular_movies = ratings['movieId'].value_counts().head(300).index
popular_titles = movies[movies['movieId'].isin(popular_movies)]['title'].sort_values().tolist()

selected_movies = st.multiselect("🎥 Film Seç:", popular_titles)

if st.button("🚀 Önerileri Göster"):
    if selected_movies:
        st.markdown("### 📚 Content-Based Öneriler:")
        cbf = content_recommendations(selected_movies)
        for title in cbf.index:
            st.write(f"🎬 {title}")

        st.markdown("### 👥 Collaborative Filtering Öneriler:")
        cf = cf_recommendations(selected_movies)
        for title in cf.index:
            st.write(f"🎬 {title}")

        st.markdown("### 🧠 Hybrid Öneriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for title in hybrid:
            st.write(f"🎬 {title}")
    else:
        st.warning("Lütfen en az bir film seç.")
