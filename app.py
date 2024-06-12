from flask import Flask, request, render_template, jsonify
import pickle
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the movie list and similarity matrix
movie_list = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('Vector_similarity.pkl', 'rb'))

# Convert the movie list DataFrame to a list of movie titles
movie_titles = movie_list['title'].tolist()

# Load the links file
links_df = pd.read_csv('links.csv')

# Function to fetch poster URL using TMDb API and movie ID
def get_poster_url(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=d6799394286d5df626eba78eb7d44bf7&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        full_path = None
    return full_path

def recommend(movie):
    index = movie_list[movie_list['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        movie_id = movie_list.iloc[i[0]].movie_id
        movie_title = movie_list.iloc[i[0]].title
        recommended_movies.append({
            'title': movie_title,
            'poster_url': get_poster_url(movie_id)
        })
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html', movie_list=movie_titles)

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    movie = request.form['movie']
    recommendations = recommend(movie)
    return render_template('index.html', recommendations=recommendations, movie_list=movie_titles, selected_movie=movie)

# Collaborative Filtering

# Load the model
try:
    model = load_model('movie_recommendation_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the user and movie encodings
with open('user2user_encoded.pkl', 'rb') as file:
    user2user_encoded = pickle.load(file)
    
with open('movie2movie_encoded.pkl', 'rb') as file:
    movie2movie_encoded = pickle.load(file)
    
with open('movie_encoded2movie.pkl', 'rb') as file:
    movie_encoded2movie = pickle.load(file)

# Load the movies dataset
movie_df = pd.read_csv('./movies.csv')

# Load the ratings dataset
ratings_df = pd.read_csv('./ratings.csv')

# Preprocess the ratings dataset
user_ids = ratings_df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = ratings_df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

ratings_df["user"] = ratings_df["userId"].map(user2user_encoded)
ratings_df["movie"] = ratings_df["movieId"].map(movie2movie_encoded)

@app.route('/personalized')
def personalized():
    return render_template('personalized.html')

@app.route('/personalized_recommend', methods=['POST'])
def personalized_recommend():
    user_id = int(request.form['user_id'])
    if user_id not in user2user_encoded:
        return render_template('personalized.html', error="User ID not found")
    
    user_encoder = user2user_encoded.get(user_id)
    
    movies_watched_by_user = ratings_df[ratings_df.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    
    movies_not_watched_index = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched_index)
    )
    print("USER MOVIE ARRAY: ",user_movie_array)

    ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]  # indices of highest 10 ratings
    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched_index[x][0]) for x in top_ratings_indices]
    
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    
    recommendations = []
    for _, row in recommended_movies.iterrows():
        tmdb_id = links_df[links_df['movieId'] == row['movieId']]['tmdbId'].values[0]
        poster_url = get_poster_url(tmdb_id)
        recommendation = {
            "title": row["title"],
            "genres": row["genres"],
            "poster_url": poster_url
        }
        recommendations.append(recommendation)
    
    return render_template('personalized.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
