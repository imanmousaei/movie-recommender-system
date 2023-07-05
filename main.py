# import gui
from rec_sys import *


if __name__ == '__main__':
    df = pd.read_csv('cleaned-dataset/pca_500.csv')
    prepare_clustering_algorithms(df)

    # gui.build_gui()