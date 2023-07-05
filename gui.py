import streamlit as st
from api import *
from rec_sys import *


def build_gui():
    st.title("Movie Recommender System")

    imdb_id = st.text_input("Enter a movie IMDB ID:")
    if imdb_id:
        try:
            
            print('imdb_id', imdb_id)
            title, year, plot, genre, awards, poster_path = get_movie_info(imdb_id)
            
            st.subheader(title)
            st.image(poster_path)
            st.write("Overview:", plot)
            st.write("Release Year:", year)
            st.write("Genre:", genre)
            st.write("Awards:", awards)
            # st.write("Vote Average:", movie_details["vote_average"])

            recommendations = get_clustering_recommendations(imdb_id, 'kmeans', num_recommendations=5)
            if recommendations:
                st.subheader("Recommendations:")
                for recommendation in recommendations:
                    title, year, plot, genre, awards, poster_path = get_movie_info(recommendation)
                    
                    st.write(title)
                    st.image(poster_path)
                    st.write("Overview:", plot)
                    st.write("Release Year:", year)
                    st.write("Genre:", genre)
                    st.write("Awards:", awards)
                    st.write("---")
            else:
                st.write("No recommendations found.")
        except ValueError:
            st.write("Invalid movie ID.")
