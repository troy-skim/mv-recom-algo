from fuzzywuzzy import process
import requests
import re

OMDB_API_KEY = "894734d2"

def fetch_omdb_poster(title):
    if not isinstance(title, str):
        raise ValueError(f"Expected a string for title, got {type(title)}")

    # Extract the movie name using regex to remove the year in parentheses
    match = re.match(r"^(.*?)(\s\(\d{4}\))?$", title)
    movie_name = match.group(1) if match else title

    # Replace spaces with '+' for the query
    formatted_title = movie_name.replace(" ", "+")

    # Build the request URL
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={formatted_title}"
    response = requests.get(url)

    # Parse the API response
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            poster_url = data.get("Poster")
            return poster_url if poster_url and poster_url != "N/A" else None
    return None  # Return None if poster is unavailable

def movie_finder(title, movies):
    all_titles = movies["title"].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0] if closest_match else None