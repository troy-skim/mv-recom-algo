from sklearn.metrics.pairwise import cosine_similarity
from module.utils import movie_finder

def compute_genre_similarity(movies):
    genres = set(genre for genre_list in movies["genres"].str.split("|") for genre in genre_list)
    for genre in genres:
        movies[genre] = movies["genres"].apply(lambda x: int(genre in x.split("|")))
    genre_matrix = movies[list(genres)].values
    return cosine_similarity(genre_matrix), genres

def recommend_by_genre(title, movies, cosine_sim, n=10):
    matched_title = movie_finder(title, movies)
    if not matched_title:
        return [f"No close matches found for '{title}'."]

    idx = movies[movies['title'] == matched_title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    recommended_titles = [movies['title'].iloc[i[0]] for i in sim_scores]
    return recommended_titles