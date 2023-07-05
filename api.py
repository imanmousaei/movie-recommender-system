import requests
import os
from secret import OMDB_API_KEY


# OMDB API configuration
BASE_URL = "http://www.omdbapi.com/"


def call_movie_api(movie_id):
    url = f"{BASE_URL}?apikey={OMDB_API_KEY}&i={movie_id}"
    response = requests.get(url)
    data = response.json()
    return data


def get_movie_poster(movie_id):
    movie_details = call_movie_api(movie_id)
    poster_url = movie_details.get("Poster")
    if poster_url != "N/A":
        response = requests.get(poster_url)
        return response.content
    None


def get_movie_info(imdb_id):
    movie_id = f"tt{imdb_id}"
    movie_details = call_movie_api(imdb_id)

    title = movie_details["Title"]
    year = movie_details["Year"]
    plot = movie_details["Plot"]
    genre = movie_details["Genre"]
    awards = movie_details["Awards"]

    poster_path = f"posters/{movie_id}.jpg"
    # if poster is already saved, don't bother downloading it
    if not os.path.exists(poster_path):
        poster_image = get_movie_poster(movie_id)
        if poster_image:
            with open(poster_path, "wb") as f:
                f.write(poster_image)
        else:
            print("No poster available.")

    return title, year, plot, genre, awards, poster_path
