from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, sparse_matrix, movie_mapper, k = 10, metric = "cosine"):
    model = NearestNeighbors(n_neighbors = k+1, algorithm = "brute", metric = metric)
    model.fit(sparse_matrix.T)

    idx = movie_mapper[movie_id]
    indices = model.kneighbors(sparse_matrix.T[idx], n_neighbors = k + 1, return_distance = False)
    
    movie_inv_mapper = {idx: movie for movie, idx in movie_mapper.items()}
    
    recommendations = [movie_inv_mapper[i] for i in indices.flatten()[1:]]

    return recommendations