import gdown
import zipfile
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix

# -------------------- VERİ YÜKLEME --------------------
@st.cache_data
def load_data():
    file_id = "1-C9k0cTqEM3Y6uHMBdwH6mGeJVDagANc"  # 32M dataset Google Drive ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "ml-latest.zip"

    # ZIP dosyasını indir
    gdown.download(url, output, quiet=False, fuzzy=True)

    # ZIP'ten çıkar
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")

    base_path = "ml-latest/"
    movies = pd.read_csv(base_path + "movies.csv")
    ratings = pd.read_csv(base_path + "ratings.csv")
    tags = pd.read_csv(base_path + "tags.csv")

    # Tag'leri birleştir
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

# -------------------- ITEM-BASED CF --------------------
@st.cache_resource
def build_item_based_model(ratings):
    movie_user_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    sparse_matrix = csr_matrix(movie_user_matrix.values)
    similarity_matrix = cosine_similarity(sparse_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    return similarity_df, movie_user_matrix

similarity_df, movie_user_matrix = build_item_based_model(ratings)

def item_based_recommendations(selected_titles, n=10):
    movie_ids = movies[movies['title'].isin(selected_titles)]['movieId'].tolist()
    similar_scores = pd.Series(dtype=float)

    for movie_id in movie_ids:
        if movie_id in similarity_df:
            sims = similarity_df[movie_id].drop(labels=movie_ids, errors='ignore')
            similar_scores = similar_scores.add(sims, fill_value=0)

    top_movie_ids = similar_scores.sort_values(ascending=False).head(n).index
    recommended_titles = movies[movies['movieId'].isin(top_movie_ids)]['title']
    return recommended_titles.tolist()

# -------------------- HYBRID MODEL --------------------
def hybrid_recommendations(selected_titles, n=10):
    cbf = content_recommendations(selected_titles, n=30)
    item_cf = item_based_recommendations(selected_titles, n=30)

    cbf_scores = pd.Series([1 - i/30 for i in range(len(cbf))], index=cbf.index)
    item_cf_scores = pd.Series([1 - i/30 for i in range(len(item_cf))], index=item_cf)

    hybrid_scores = cbf_scores.add(item_cf_scores, fill_value=0)
    hybrid_scores = hybrid_scores.sort_values(ascending=False).head(n)

    return hybrid_scores.index.tolist()

# -------------------- STREAMLIT UI --------------------
st.title("🎬 Film Öneri Sistemi")
st.markdown("Beğendiğin filmleri seç, sistem senin için öneri yapsın.")

st.markdown(f"""
#### 📊 Sistem Hakkında
Bu öneriler, **{len(movies):,} film** ve **{len(ratings):,} kullanıcı oyu** temel alınarak oluşturulmuştur.  
Hem içerik benzerliği hem de izleyici davranışları birlikte analiz edilerek en uygun filmler sunulmaktadır.
""")

popular_movies = ratings['movieId'].value_counts().head(300).index
popular_titles = movies[movies['movieId'].isin(popular_movies)]['title'].sort_values().tolist()

selected_movies = st.multiselect("🎥 Film Seç:", popular_titles)

if st.button("🚀 Önerileri Göster"):
    if selected_movies:
        st.markdown("### 📚 Content-Based Öneriler:")
        cbf = content_recommendations(selected_movies)
        for title in cbf.index:
            st.write(f"🎬 {title}")

        st.markdown("### 🧩 Item-Based CF Öneriler:")
        item_cf = item_based_recommendations(selected_movies)
        for title in item_cf:
            st.write(f"🎬 {title}")

        st.markdown("### 🧠 Hybrid Öneriler:")
        hybrid = hybrid_recommendations(selected_movies)
        for title in hybrid:
            st.write(f"🎬 {title}")
    else:
        st.warning("Lütfen en az bir film seç.")
