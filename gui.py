import streamlit as st
import requests

# TMDB API configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.themoviedb.org/3"

# Function to get movie recommendations
def get_movie_recommendations(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["results"]

# Function to get movie details
def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "language": "en-US"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Streamlit app
def build_gui():
    st.title("Movie Recommender System")

    # Movie input
    movie_id = st.text_input("Enter a movie ID:")
    if movie_id:
        try:
            movie_id = int(movie_id)
            movie_details = get_movie_details(movie_id)
            st.subheader(movie_details["title"])
            st.image(f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}")
            st.write("Overview:", movie_details["overview"])
            st.write("Release Date:", movie_details["release_date"])
            st.write("Vote Average:", movie_details["vote_average"])

            recommendations = get_movie_recommendations(movie_id)
            if recommendations:
                st.subheader("Recommendations:")
                for recommendation in recommendations:
                    st.write(recommendation["title"])
                    st.image(f"https://image.tmdb.org/t/p/w500{recommendation['poster_path']}")
                    st.write("Overview:", recommendation["overview"])
                    st.write("Release Date:", recommendation["release_date"])
                    st.write("Vote Average:", recommendation["vote_average"])
                    st.write("---")
            else:
                st.write("No recommendations found.")
        except ValueError:
            st.write("Invalid movie ID.")

