import pandas as pd
from scipy.sparse import csr_matrix

def load_data(ratings_path, movies_path):
    # read from csvs into dataframes
    # return ratings dataframe, movies dataframe
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def create_sparse_matrix(ratings):
    # generate sparse matrix from ratings dataframe
    # return matrix, user mapping dictionary, movie mapping dictionary
    user_mapper = {user: idx for idx, user in enumerate(ratings["userId"].unique())}
    movie_mapper = {movie: idx for idx, movie in enumerate(ratings["movieId"].unique())}

    user_idx = ratings["userId"].map(user_mapper)
    movie_idx = ratings["movieId"].map(movie_mapper)

    sparse_matrix = csr_matrix((ratings["rating"], (user_idx, movie_idx)))

    return sparse_matrix, user_mapper, movie_mapper
