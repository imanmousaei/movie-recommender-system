import pandas as pd


def movie_id_to_imdb_id(movie_id):
    links = pd.read_csv('./movies-dataset/links.csv')

    imdb_id = links.loc[links['movieId'] == movie_id, 'imdbId'].values[0]

    return imdb_id
