import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, MeanShift, Birch
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
# import recmetrics


n_clusters = 10

clustering_algorithms = {
    'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'),
    "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
    'DBSCAN': DBSCAN(eps=2, min_samples=5),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters, n_init="auto"),
    'MeanShift': MeanShift(bin_seeding=True),
    'Birch': Birch(n_clusters=n_clusters),
}


def prepare_clustering_algorithms():
    df = pd.read_csv('cleaned-dataset/pca_500.csv')

    for name, algorithm in clustering_algorithms.items():
        name = name.lower()
        filename = f'clustering/{name}.csv'
        if os.path.exists(filename):
            continue

        data = df.drop(['id', 'imdb_id'], axis=1)

        algorithm.fit(data)
        if hasattr(algorithm, "labels_"):
            clusters = algorithm.labels_.astype(int)
        else:
            clusters = algorithm.predict(data)

        df['cluster'] = clusters
        df.to_csv(filename)
        print(f'{name} finished')


def get_clustering_recommendations(imdb_id, algorithm_name, num_recommendations):
    algorithm_name = algorithm_name.lower()
    filename = f'clustering/{algorithm_name}.csv'

    # todo: handle movie collections

    df = pd.read_csv(filename)

    movie_cluster = df.loc[df['imdb_id'] == imdb_id, 'cluster'].values[0]
    similar_movies = df[df['cluster'] == movie_cluster]
    similar_movies = similar_movies[similar_movies.index != imdb_id]
    similar_movies = similar_movies.sample(num_recommendations)

    return similar_movies['imdb_id'].values


def get_closest_movies(imdb_id):
    # Create an instance of KNN algorithm
    knn = NearestNeighbors(n_neighbors=n)

    # Fit the data to KNN algorithm
    knn.fit(movie_data)

    # Calculate distances and indices of nearest neighbors for a given movie
    distances, indices = n.kneighbors(query_movie)

    # Get the n closest movies
    closest_movies = movie_data.iloc[indices[0]]


def get_closest_cosine_similarity(movie_id):
    # Assuming you have a movie_feature_matrix as your dataset
    similarity_matrix = np.dot(A, A)/(norm(A)*norm(A))

    # Assuming given_movie_index is the index of the given movie
    sorted_indices = np.argsort(similarity_matrix[given_movie_index])[::-1]

    # Assuming n is the number of closest movies you want to retrieve
    closest_movie_indices = sorted_indices[1:n+1]  # Exclude given movie index

    closest_movies = movie_dataset.iloc[closest_movie_indices]
