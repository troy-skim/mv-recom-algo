from flask import Flask, request, jsonify, render_template
from module.data_prep import load_data, create_sparse_matrix
from module.collaborative import find_similar_movies
from module.content_based import compute_genre_similarity, recommend_by_genre
from module.utils import movie_finder, fetch_omdb_poster
import numpy as np
import os

app = Flask(__name__)

ratings, movies = load_data("/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/ratings.csv", "/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/movies.csv")
sparse_matrix, user_mapper, movie_mapper, movieId_to_title, movieTitle_to_Id = create_sparse_matrix(ratings, movies)
cosine_sim, genres = compute_genre_similarity(movies)

user_likes = []
user_dislikes = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/movies", methods=["GET"])
def get_random_movies():
    # Number of movies to show (default 10)
    num_movies = int(request.args.get("num", 10))
    random_movies = movies.sample(num_movies)
    movie_list = random_movies[["movieId", "title"]].to_dict(orient="records")
    return jsonify(movie_list)

@app.route("/feedback", methods=["POST"])
def record_feedback():
    data = request.json
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

@app.route("/api/recommend", methods=["GET"])
def api_recommend_movies():
    if not user_likes:
        return jsonify({"error": "No liked movies to base recommendations on."}), 400

    recommendations = set()

    for movie_id in user_likes:
        try:
            # Collaborative filtering
            collaborative_recs = find_similar_movies(
                movie_id=movie_id,
                sparse_matrix=sparse_matrix,
                movie_mapper=movie_mapper,
                k=5,
                metric="cosine"
            )
            recommendations.update(collaborative_recs)

            # Content-based filtering
            movie_name = movies[movies['movieId'] == movie_id]['title'].iloc[0]
            content_recs = recommend_by_genre(
                title=movie_name,
                movies=movies,
                cosine_sim=cosine_sim,
                n=5
            )
            recommendations.update(content_recs)
        except Exception as e:
            print(f"Error processing recommendations for movie ID {movie_id}: {e}")

    final_recommendations = [
        movieId_to_title[rec] if rec in movieId_to_title else rec
        for rec in recommendations
    ]

    # Exclude disliked movies
    final_recommendations = [
        rec for rec in final_recommendations if rec not in user_dislikes
    ]

    final_recommendations = [
        {"title": rec, "poster_url": fetch_omdb_poster(rec)} for rec in final_recommendations
    ]

    if not final_recommendations:
        return jsonify({"error": "No recommendations could be generated."}), 400

    return jsonify(final_recommendations)

@app.route("/recommend", methods=["GET"])
def recommend_movies_page():
    response = api_recommend_movies()
    if response.status_code == 400:
        recommendations = []
    else:
        recommendations = response.get_json()

    return render_template("personalized_results.html", recommendations=recommendations)

@app.route("/results", methods=["GET"])
def results():
    rec_type = request.args.get("type")
    recommendations = []

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
    return render_template("results.html", recommendations=recommendations, title = title)
    
if __name__ == "__main__":
    app.run(port = 8000, debug=True)