import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, KNNBasic

# -------------------- VERİ YÜKLEME --------------------

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

# -------------------- COLLABORATIVE FILTERING MODEL --------------------

@st.cache_resource
def train_cf_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    sim_options = {
        'name': 'cosine',
        'user_based': False  # Item-based CF
    }

    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model, trainset

cf_model, trainset = train_cf_model(ratings)

def cf_recommendations(selected_titles, n=10):
    movie_ids = movies[movies['title'].isin(selected_titles)]['movieId'].tolist()
    all_scores = {}

    for movie_id in movie_ids:
        if movie_id not in trainset._raw2inner_id_items:
            continue
        inner_id = trainset.to_inner_iid(movie_id)
        neighbors = cf_model.get_neighbors(inner_id, k=n+20)
        for neighbor_inner_id in neighbors:
            raw_id = trainset.to_raw_iid(neighbor_inner_id)
            title_row = movies[movies['movieId'] == int(raw_id)]
            if not title_row.empty:
                title = title_row['title'].values[0]
                if title not in selected_titles:
                    all_scores[title] = all_scores.get(title, 0) + 1

    return pd.Series(all_scores).sort_values(ascending=False).head(n)

# -------------------- HYBRID MODEL --------------------

def hybrid_recommendations(selected_titles, n=10):
    cbf_scores = content_recommendations(selected_titles, n=30)
    cf_scores = cf_recommendations(selected_titles, n=30)

    # Normalize skorlar (0-1 arası) + eksiklere 0 ver
    cbf_scores = (cbf_scores.max() - cbf_scores) / (cbf_scores.max() - cbf_scores.min() + 1e-6)
    cf_scores = (cf_scores.max() - cf_scores) / (cf_scores.max() - cf_scores.min() + 1e-6)

    cbf_scores = cbf_scores / cbf_scores.sum()
    cf_scores = cf_scores / cf_scores.sum()

    hybrid = (cbf_scores * 0.5).add(cf_scores * 0.5, fill_value=0)
    return hybrid.sort_values(ascending=False).head(n)

# -------------------- STREAMLIT UI --------------------

st.title("🎬 Film Öneri Sistemi")
st.write("İzlediğin filmleri seç, 3 farklı öneri sisteminden tavsiye al!")

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
        for title in cf.index:
            st.write(f"🎬 {title}")

        st.subheader("🧠 Hybrid Öneriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for title in hybrid.index:
            st.write(f"🎬 {title}")
    else:
        st.warning("Lütfen en az bir film seç.")
