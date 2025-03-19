import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setlerini yükleyelim
movies_df = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/movies.csv")
ratings_df = pd.read_csv("raw.githubusercontent.com/kleax/filmrecommendation/main/ratings.csv")

# Türleri işleyelim (Content-based)
movies_df['genres'] = movies_df['genres'].fillna('')
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

# Cosine benzerliği hesaplayalım
genre_similarity = cosine_similarity(genre_matrix)
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies_df['title'], columns=movies_df['title'])

# Popüler filmleri bulalım
movie_ratings_clean = pd.merge(ratings_df, movies_df, on='movieId').drop('timestamp', axis=1)
popular_movies = movie_ratings_clean.groupby('title').rating.count().sort_values(ascending=False).head(30).index.tolist()

# Streamlit UI
st.title('🎬 Film Öneri Sistemi')

selected_movies = st.multiselect("Beğendiğin Filmleri Seç:", popular_movies)

year_range = st.slider('Hangi yıllar arasında filmler önerelim?', 1950, 2020, (2000, 2010))

def recommend_movies_content(selected_movies, similarity_df, movies_df, year_range, n=5):
    similarity_scores = similarity_df[selected_movies].mean(axis=1)
    similarity_scores = similarity_scores.drop(labels=selected_movies)
    recommendations = similarity_scores.sort_values(ascending=False).reset_index()

    movies_df['year'] = movies_df['title'].str.extract('\((\d{4})\)').astype(float)
    filtered_movies = movies_df[(movies_df['year'] >= year_range[0]) & (movies_df['year'] <= year_range[1])]

    recommendations = recommendations[recommendations['title'].isin(filtered_movies['title'])].head(n)
    return recommendations['title'].tolist()

if st.button('Önerileri Göster'):
    if selected_movies:
        recommendations = recommend_movies_content(selected_movies, genre_similarity_df, movies_df, year_range)
        st.subheader('Sana özel öneriler:')
        for movie in recommendations:
            st.write(f'🎯 {movie}')
    else:
        st.warning('Lütfen önce film seçimi yap.')
