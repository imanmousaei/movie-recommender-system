import gradio as gr
from api import *
from rec_sys import *
from PIL import Image

global_imdb_id = None
global_recommendations = []

def recommender(imdb_id, recommendation_id):
    global global_imdb_id
    global global_recommendations
    
    if imdb_id != global_imdb_id:
        global_recommendations = []
        global_imdb_id = imdb_id
    
    title, year, plot, genre, awards, poster_path = get_movie_info(imdb_id)
    if recommendation_id == None:
        poster = Image.open(poster_path)
        return title, year, plot, genre, awards, poster
    
    print('recommendation_id', recommendation_id)
    recommendation_id = int(recommendation_id) - 1
    
    if len(global_recommendations) == 0:
        num_recommendations = 5
        recommendations = get_clustering_recommendations(imdb_id, 'dbscan', num_recommendations=num_recommendations)
        global_recommendations = recommendations
        
    print('global_recommendations', global_recommendations)
    recom_title, recom_year, recom_plot, recom_genre, recom_awards, recom_poster_path = get_movie_info(global_recommendations[recommendation_id])
    
    filepath = './posters/tt0028667.jpg'
    recom_poster = Image.open(recom_poster_path)
    
    return recom_title, recom_year, recom_plot, recom_genre, recom_awards, recom_poster
   
def get_labeled_textbox(label):
    return gr.outputs.Textbox(label=label)

def build_gradio_gui():
    main_ui = gr.Interface(
        fn=recommender,
        inputs = [
            'text',
            gr.Radio(["1", "2", "3", "4", "5"]),
        ],
        outputs = [
            get_labeled_textbox('Title'),
            # 'number',
            get_labeled_textbox('Production Year'),
            get_labeled_textbox('Plot'),
            get_labeled_textbox('Genre'),
            get_labeled_textbox('Awards'),
            'image', 
        ],
        live=True,
        title='Movie Recommendation System'
    )
    
    main_ui.launch()
