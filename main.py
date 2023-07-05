# import gui
from rec_sys import *


if __name__ == '__main__':
    # df = pd.read_csv('cleaned-dataset/pca_500.csv')
    # prepare_clustering_algorithms()
    
    movie_id = 862
    recom_ids = get_clustering_recommendations('kmeans', 5, movie_id)
    print('recom_ids', recom_ids)
    

    # gui.build_gui()