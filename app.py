from flask import Flask, request, jsonify, render_template
from module.data_prep import load_data, create_sparse_matrix
from module.collaborative import find_similar_movies
from module.content_based import compute_genre_similarity, recommend_by_genre
from module.utils import movie_finder
import os

app = Flask(__name__)

ratings, movies = load_data("/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/ratings.csv", "/Users/troy_skim/Desktop/cs_projects/mv-recom-algo/data/raw/ml-latest-small/movies.csv")
sparse_matrix, user_mapper, movie_mapper, movieId_to_title, movieTitle_to_Id = create_sparse_matrix(ratings, movies)
cosine_sim, genres = compute_genre_similarity(movies)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results", methods=["GET"])
def results():
    rec_type = request.args.get("type")
    recommendations = []

    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Movie title is required"}), 400
    # collaborative part
    movie_id = movieTitle_to_Id[movie_finder(title, movies)]
    recommendations = find_similar_movies(movie_id, sparse_matrix, movie_mapper, k=10)
    recommendations = [movieId_to_title[movieId] for movieId in recommendations]
    recommendations += recommend_by_genre(title, movies, cosine_sim, n=10)
    recommendations = list(set(recommendations))
    return render_template("results.html", recommendations=recommendations)
    
if __name__ == "__main__":
    app.run(debug=True)
