from flask import Flask, request, jsonify, render_template
from module.data_prep import load_data, create_sparse_matrix
from module.collaborative import find_similar_movies
from module.content_based import compute_genre_similarity, recommend_by_genre
from module.utils import movie_finder, fetch_omdb_poster
import numpy as np
import os
import random

app = Flask(__name__)

# Load the ratings and movies data from CSV files
DATA_DIR = os.getenv("MOVIELENS_DATA_DIR", "data/raw/ml-latest-small")
ratings, movies = load_data(
    os.path.join(DATA_DIR, "ratings.csv"),
    os.path.join(DATA_DIR, "movies.csv"),
)

# Create a sparse matrix for collaborative filtering and mapping dictionaries
sparse_matrix, user_mapper, movie_mapper, movieId_to_title, movieTitle_to_Id = create_sparse_matrix(ratings, movies)

# Compute the cosine similarity matrix for content-based filtering
cosine_sim, genres = compute_genre_similarity(movies)

# Global lists to track user preferences (likes and dislikes)
user_likes = []  # Movies liked by the user
user_dislikes = []  # Movies disliked by the user

# Helper function to generate recommendations and apply filters
def get_filtered_recommendations(limit=10):
    """
    Generate movie recommendations by combining collaborative and content-based filtering.
    Shuffle the recommendations to ensure diversity and exclude disliked movies.
    
    Parameters:
    limit (int): Number of recommendations to return.

    Returns:
    list: A list of dictionaries containing movie titles and poster URLs.
    """
    if not user_likes:
        return None  # Return None if no liked movies to base recommendations on

    recommendations = set()  # Use a set to avoid duplicate recommendations

    for movie_id in user_likes:
        try:
            # Collaborative filtering: find similar movies based on user preferences
            collaborative_recs = find_similar_movies(
                movie_id=movie_id,
                sparse_matrix=sparse_matrix,
                movie_mapper=movie_mapper,
                k=15,  # Fetch more recommendations to introduce variety
                metric="cosine"
            )
            recommendations.update(collaborative_recs)

            # Content-based filtering: find movies with similar genres
            movie_name = movies[movies['movieId'] == movie_id]['title'].iloc[0]
            content_recs = recommend_by_genre(
                title=movie_name,
                movies=movies,
                cosine_sim=cosine_sim,
                n=15  # Fetch more recommendations for shuffling
            )
            recommendations.update(content_recs)
        except Exception as e:
            print(f"Error processing recommendations for movie ID {movie_id}: {e}")

    # Convert movie IDs to titles and exclude disliked movies
    final_recommendations = [
        movieId_to_title[rec] if rec in movieId_to_title else rec
        for rec in recommendations if rec not in user_dislikes
    ]

    # Shuffle the recommendations to add randomness
    random.shuffle(final_recommendations)

    # Limit the number of recommendations returned
    final_recommendations = final_recommendations[:limit]

    # Add poster URLs to the recommendations
    final_recommendations = [
        {"title": rec, "poster_url": fetch_omdb_poster(rec)} for rec in final_recommendations
    ]

    return final_recommendations

# Route for the home page
@app.route("/")
def home():
    """
    Render the homepage (index.html) where the user can search for a movie,
    see random movies, and interact with "like" and "dislike" buttons.
    """
    return render_template("index.html")

# Route to fetch random movies for discovery
@app.route("/movies", methods=["GET"])
def get_random_movies():
    """
    Return a list of randomly selected movies with their poster URLs.
    
    Query Parameters:
    num (int): Number of random movies to return (default: 12).
    
    Returns:
    JSON: A list of dictionaries containing movie IDs, titles, and poster URLs.
    """
    num_movies = int(request.args.get("num", 12))  # Number of movies to show (default 12)
    random_movies = movies.sample(num_movies)

    movie_list = []  # Store movie details with poster URLs
    for _, movie in random_movies.iterrows():
        poster_url = fetch_omdb_poster(movie["title"])  # Fetch poster URL from OMDb API
        movie_list.append({
            "movieId": movie["movieId"],
            "title": movie["title"],
            "poster_url": poster_url or "/static/images/default-poster.jpg",  # Fallback poster image
        })

    return jsonify(movie_list)

# Route to record user feedback (like, dislike, or remove)
@app.route("/feedback", methods=["POST"])
def record_feedback():
    """
    Record user feedback (like, dislike, or remove) for a movie.
    
    Request Body:
    {
        "movieId": int,
        "action": str ("like", "dislike", or "remove")
    }
    
    Returns:
    JSON: Updated lists of liked and disliked movies.
    """
    data = request.json  # Parse the JSON request body
    movie_id = data.get("movieId")
    action = data.get("action")

    if not movie_id or action not in ["like", "dislike", "remove"]:
        return jsonify({"error": "Invalid input"}), 400

    if action == "like":
        if movie_id not in user_likes:
            user_likes.append(movie_id)
        if movie_id in user_dislikes:
            user_dislikes.remove(movie_id)
    elif action == "dislike":
        if movie_id not in user_dislikes:
            user_dislikes.append(movie_id)
        if movie_id in user_likes:
            user_likes.remove(movie_id)
    elif action == "remove":
        if movie_id in user_likes:
            user_likes.remove(movie_id)
        if movie_id in user_dislikes:
            user_dislikes.remove(movie_id)

    print(f"Updated preferences -> likes: {user_likes} dislikes: {user_dislikes}")
    return jsonify({"message": "Feedback recorded", "likes": user_likes, "dislikes": user_dislikes})

# API route to return recommendations as JSON
@app.route("/api/recommend", methods=["GET"])
def api_recommend_movies():
    """
    Return personalized movie recommendations as JSON.
    
    Query Parameters:
    limit (int): Number of recommendations to return (default: 10).
    
    Returns:
    JSON: A list of recommended movies with titles and poster URLs.
    """
    limit = int(request.args.get("limit", 10))
    final_recommendations = get_filtered_recommendations(limit)

    if not final_recommendations:
        return jsonify({"error": "No recommendations could be generated."}), 400

    return jsonify(final_recommendations)

# Route to render personalized recommendations as an HTML page
@app.route("/recommend", methods=["GET"])
def recommend_movies_page():
    """
    Render the personalized recommendations page using "personalized_results.html".
    """
    limit = int(request.args.get("limit", 10))
    final_recommendations = get_filtered_recommendations(limit)

    if not final_recommendations:
        return render_template("personalized_results.html", recommendations=[])

    return render_template("personalized_results.html", recommendations=final_recommendations)

# Route to get recommendations based on a searched movie title
@app.route("/results", methods=["GET"])
def results():
    """
    Return recommendations based on the movie title entered by the user.
    
    Query Parameters:
    title (str): The title of the movie to base recommendations on.
    
    Returns:
    HTML: Rendered template showing recommendations.
    """
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Movie title is required"}), 400

    movie_id = movieTitle_to_Id[movie_finder(title, movies)]
    recommendations = find_similar_movies(movie_id, sparse_matrix, movie_mapper, k=10)
    recommendations = [movieId_to_title[movieId] for movieId in recommendations]
    recommendations += recommend_by_genre(title, movies, cosine_sim, n=10)
    recommendations = list(set(recommendations))
    recommendations = [
        {"title": rec, "poster_url": fetch_omdb_poster(rec)} for rec in recommendations
    ]

    return render_template("results.html", recommendations=recommendations, title=title)

# Run the Flask app
if __name__ == "__main__":
    app.run(port=8000, debug=True)
