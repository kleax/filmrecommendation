import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    movies_df = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/movies.csv")
    ratings_df = pd.read_csv("https://raw.githubusercontent.com/kleax/filmrecommendation/refs/heads/main/ratings.csv")
    return movies_df, ratings_df

movies_df, ratings_df = load_data()
#new
@st.cache_data
def calculate_genre_similarity(movies_df):
    vectorizer = CountVectorizer(token_pattern=r'[^|]+')
    genre_matrix = vectorizer.fit_transform(movies_df['genres'])
    genre_similarity = cosine_similarity(genre_matrix)
    return pd.DataFrame(genre_similarity, index=movies_df['title'], columns=movies_df['title'])
#new

# Genre-based similarity matrix
movies_df['genres'] = movies_df['genres'].fillna('')
"""vectorizer = CountVectorizer(token_pattern=r'[^|]+')
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

genre_similarity = cosine_similarity(genre_matrix)"""
genre_similarity_df = calculate_genre_similarity(movies_df)

# beklemede genre_similarity_df = pd.DataFrame(genre_similarity, index=movies_df['title'], columns=movies_df['title'])

# Popular movies selection
movie_ratings_clean = pd.merge(ratings_df, movies_df, on='movieId').drop(columns=['timestamp'])
popular_movies = movie_ratings_clean.groupby('title')['rating'].count().sort_values(ascending=False).head(50).index.tolist()
popular_movies.sort()

st.title('🎬 Film Öneri Sistemi')

selected_movies = st.multiselect("Beğendiğin Filmleri Seç:", popular_movies)
year_range = st.slider('Hangi yıllar arasında filmler önerelim?', 1950, 2020, (2000, 2010))

@st.cache_data
def recommend_movies_content(selected_movies, similarity_df, movies_df, year_range, n=5):
    if not selected_movies:
        return []

    # Filter similarity matrix to only include selected movies that exist in the index
    selected_movies = [movie for movie in selected_movies if movie in similarity_df.index]

    if not selected_movies:
        return []

    # Compute similarity scores
    similarity_scores = similarity_df.loc[selected_movies].mean(axis=0).drop(labels=selected_movies)
    recommendations = similarity_scores.sort_values(ascending=False).reset_index()
    recommendations.columns = ['title', 'similarity']

    # Extract movie year
    movies_df = movies_df.copy()
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_df = movies_df.dropna(subset=['year'])
    movies_df['year'] = movies_df['year'].astype(int)

    # Filter by year
    filtered_movies = movies_df[(movies_df['year'] >= year_range[0]) & (movies_df['year'] <= year_range[1])]
    recommendations = recommendations[recommendations['title'].isin(filtered_movies['title'])].head(n)

    return recommendations['title'].tolist()

if st.button('Önerileri Göster'):
    if selected_movies:
        recommendations = recommend_movies_content(selected_movies, genre_similarity_df, movies_df, year_range)
        if recommendations:
            st.subheader('Sana özel öneriler:')
            for movie in recommendations:
                st.write(f'🎯 {movie}')
        else:
            st.warning('Seçilen yıl aralığında önerilebilecek film bulunamadı.')
    else:
        st.warning('Lütfen önce film seçimi yap.')
