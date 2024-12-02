from flask import Flask, request, jsonify, render_template
from module.data_prep import load_data, create_sparse_matrix
from module.collaborative import find_similar_movies
from module.content_based import compute_genre_similarity, recommend_by_genre
from module.utils import movie_finder
import os

app = Flask(__name__)

ratings, movies = load_data("/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/ratings.csv", "/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/movies.csv")
sparse_matrix, user_mapper, movie_mapper = create_sparse_matrix(ratings)
cosine_sim, genres = compute_genre_similarity(movies)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results", methods=["GET"])
def results():
    rec_type = request.args.get("type")
    recommendations = []

    if rec_type == "collaborative":
        movie_id = request.args.get("movie_id")
        if not movie_id:
            return "Movie ID is required for collaborative filtering."
        try:
            movie_id = int(movie_id)
            recommendations = find_similar_movies(movie_id, sparse_matrix, movie_mapper, k=10)
        except ValueError:
            return "Invalid Movie ID."

    elif rec_type == "content":
        title = request.args.get("title")
        if not title:
            return jsonify({"error": "Movie title is required"}), 400
        recommendations = recommend_by_genre(title, movies, cosine_sim, n=10)

    return render_template("results.html", recommendations=recommendations)
    
if __name__ == "__main__":
    app.run(debug=True)
