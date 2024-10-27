import streamlit as st
import pandas as pd

# Load your model and data
with open('66130700307recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Function to get movie recommendations
def get_recommendations(user_id, num_recommendations=5):
    user_ratings = movie_ratings[movie_ratings['userId'] == user_id]
    user_unrated_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]
    user_unrated_movies['est_rating'] = user_unrated_movies['movieId'].apply(lambda x: svd_model.predict(user_id, x).est)
    recommendations = user_unrated_movies.sort_values('est_rating', ascending=False).head(num_recommendations)
    return recommendations

# Streamlit app
st.title('Movie Recommender System')

user_id = st.number_input('Enter User ID', min_value=1, max_value=movie_ratings['userId'].max())
num_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=20, value=5)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_id, num_recommendations)
    st.write(recommendations)


