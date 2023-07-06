import gradio as gr
import streamlit as st
from api import *
from rec_sys import *


def build_st_gui():
    st.set_page_config(layout="wide")
    st.title("Movie Recommender System")

    imdb_id = st.text_input("Enter a movie IMDB ID:")
    if imdb_id:
        print('imdb_id', imdb_id)
        your_title, your_year, your_plot, your_genre, your_awards, your_poster_path = get_movie_info(
            imdb_id)

        st.subheader(your_title)
        st.image(your_poster_path)
        st.write("Overview:", your_plot)
        st.write("Release Year:", your_year)
        st.write("Genre:", your_genre)
        st.write("Awards:", your_awards)
        # st.write("Vote Average:", movie_details["vote_average"])

        num_recommendations = 5
        recommendations = get_clustering_recommendations(
            imdb_id, 'kmeans', num_recommendations=num_recommendations)

        titles, years, plots, genres, awardss, poster_paths = [], [], [], [], [], []
        # titles.append(your_title)
        # years.append(your_year)
        # plots.append(your_plot)
        # genres.append(your_genre)
        # awardss.append(your_awards)
        # poster_paths.append(your_poster_path)

        print('recommendations', recommendations)

        if len(recommendations) >= 1:
            st.subheader("Recommendations:")
            for recommendation in recommendations:
                title, year, plot, genre, awards, poster_path = get_movie_info(
                    recommendation)
                titles.append(title)
                years.append(year)
                plots.append(plot)
                genres.append(genre)
                awardss.append(awards)
                poster_paths.append(poster_path)
        else:
            st.write("No recommendations found.")

        menu_items = titles
        selected_item = st.sidebar.selectbox('Recommendations', menu_items)

        selected_idx = menu_items.index(selected_item)
        print('selected_item', selected_item)
        print('selected_idx', selected_idx)

        st.subheader(titles[selected_idx])
        st.image(poster_paths[selected_idx])
        st.write("Overview:", plots[selected_idx])
        st.write("Release Year:", years[selected_idx])
        st.write("Genre:", genres[selected_idx])
        st.write("Awards:", awardss[selected_idx])
        st.write("---")


def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2

def build_gradio_gui():
#     sidebar = gr.Interface(
#         fn=None,
#         inputs='text',
#         outputs=None,
#         titleSidebar="",
#         layout="vertical",
#         description="Select page a",
#         examples=None,
#         default=" theme",
#    )

#     sidebar.add_button("Page 1", item1)
#     sidebar.add_button("Page 2", item2)

    image = gr.Image(shape=(224, 224))
    label = gr.Label(num_top_classes=3)
   
    demo = gr.Interface(
        calculator,
        [
            "number",
            gr.Radio(["add", "subtract", "multiply", "divide"]),
            "number"
        ],
        "number",
        live=True,
    )
    demo.launch()
