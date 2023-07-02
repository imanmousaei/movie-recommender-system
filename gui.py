import streamlit as st



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

