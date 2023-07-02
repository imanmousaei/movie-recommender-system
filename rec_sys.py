import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import recmetrics


# Generate movie features matrix
movies = np.array([[1, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1],
                   [0, 0, 1, 0, 1],
                   [1, 1, 0, 1, 0]])

# Cluster-based methods
methods = {
    "K-means": KMeans(n_clusters=2),
    "Hierarchical": AgglomerativeClustering(n_clusters=2),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=2)
}

# Evaluate each method using Intra-list Similarity
for method_name, method in methods.items():
    # Fit the clustering model
    clusters = method.fit_predict(movies)

    # Calculate Intra-list Similarity
    intra_list_similarity = recmetrics.intra_list_similarity(clusters, movies)

    # Print the results
    print(f"{method_name} Intra-list Similarity:", intra_list_similarity)
