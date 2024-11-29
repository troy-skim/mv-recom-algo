from sklearn.metrics.pairwise import consine_similarity

def compute_genre_similarity(movies):
    genres = set(genre for genre_list in movies["genres"].str.split("|") for genre in genre_list)
    for genre in genres:
        movies[genre] = movies["genres"].apply(lambda x: int(genre in x.split("|")))
    genre_matrix = movies[list(genres)].values
    return consine_similarity(genre_matrix), genres

def recommend_by_genre(title, movies, cosine_sim, n = 10):
    idx = movies[movies["title"] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key = lambda x: x[1], reverse = True)[1:n+1]
    recommended_titles = movies["title"].iloc[[i[0] for i in sim_scores]].tolist()
    return recommended_titles
