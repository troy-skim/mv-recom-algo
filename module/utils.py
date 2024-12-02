from fuzzywuzzy import process

def movie_finder(title, movies):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0] if closest_match else None