import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 📦 Veri yükleme
@st.cache_data
def load_data():
    movies = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/movies.csv")
    ratings = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/ratings.csv")
    tags = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/tags.csv")

    # 🎯 Tag'leri birleştir
    tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tags_agg, on='movieId', how='left')
    
    # 🎯 Content sütunu oluştur
    movies['content'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['tag'].fillna('')
    
    return movies, ratings

movies, ratings = load_data()

# 🎯 TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title'])

# ✅ Content-Based
def content_recommendations(selected_titles, n=10):
    valid_titles = [t for t in selected_titles if t in indices]
    if not valid_titles:
        return []
    all_scores = pd.Series(dtype=float)
    for title in valid_titles:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+20]
        for i, score in sim_scores:
            all_scores[movies['title'].iloc[i]] = all_scores.get(movies['title'].iloc[i], 0) + score
    all_scores = all_scores.drop(labels=valid_titles, errors="ignore").sort_values(ascending=False)
    return list(all_scores.head(n).index)

# ✅ Collaborative Filtering
def cf_from_movies(titles, n=10, min_rating=4.0):
    movie_ids = movies[movies['title'].isin(titles)]['movieId'].tolist()
    users_who_liked = ratings[(ratings['movieId'].isin(movie_ids)) & (ratings['rating'] >= min_rating)]['userId'].unique()
    similar_ratings = ratings[(ratings['userId'].isin(users_who_liked)) & (~ratings['movieId'].isin(movie_ids))]
    recommendation_scores = similar_ratings.groupby('movieId')['rating'].mean()
    top_movie_ids = recommendation_scores.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()

# ✅ Hybrid
def hybrid_recommendations(titles, n=10):
    cbf = content_recommendations(titles, n=30)
    cf = cf_from_movies(titles, n=30)

    cbf_scores = pd.Series([1 - i/30 for i in range(len(cbf))], index=cbf)
    cf_scores = pd.Series([1 - i/30 for i in range(len(cf))], index=cf)
    
    hybrid_scores = cbf_scores.add(cf_scores, fill_value=0).sort_values(ascending=False)
    return hybrid_scores.head(n).index.tolist()

# ----------------- STREAMLIT UI -----------------

st.title("🎬 Film Öneri Sistemi (CBF + CF + Hybrid)")

popular_movies = ratings.groupby('movieId').size().sort_values(ascending=False).head(300).index
popular_titles = movies[movies['movieId'].isin(popular_movies)]['title'].sort_values()

selected_movies = st.multiselect("Beğendiğin filmleri seç:", popular_titles)

if st.button("🎯 Önerileri Göster"):
    if selected_movies:
        st.subheader("📚 Content-Based Öneriler:")
        cbf = content_recommendations(selected_movies)
        for movie in cbf:
            st.write(f"🎬 {movie}")

        st.subheader("👥 Collaborative Filtering Öneriler:")
        cf = cf_from_movies(selected_movies)
        for movie in cf:
            st.write(f"🎬 {movie}")

        st.subheader("🧠 Hybrid Öneriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for movie in hybrid:
            st.write(f"🎬 {movie}")
    else:
        st.warning("Lütfen en az bir film seç.")
