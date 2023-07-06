from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from rec_sys import clustering_algorithms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def compute_infra_list_similarity(df, algorithm_name):
    features = df.drop(['cluster', 'id', 'imdb_id'], axis=1)
    predicted_labels = df['cluster'].values
    # print('predicted_labels', predicted_labels[:75])
    
    # predicted_labels = [x if x != 0 else 10 for x in predicted_labels]  # replace 0 with 10

    # intra_list_similarity = recmetrics.intra_list_similarity(predicted_labels, features)
    # print('intra_list_similarity', intra_list_similarity)
    
    intra_list_similarity = {}
    unique_clusters = df['cluster'].unique()

    for cluster in unique_clusters:
        cluster_features = features[predicted_labels == cluster]
        distance_matrix = pairwise_distances(cluster_features)
        dist = (distance_matrix.sum() / (distance_matrix.shape[0] * (distance_matrix.shape[0] - 1)))
        intra_list_similarity[cluster] = dist
        # print(f'Cluster {cluster} Distance:', dist)

    keys = list(intra_list_similarity.keys())
    values = list(intra_list_similarity.values())
    plt.clf()
    plt.bar(keys, values)
    plt.title(f'Intra List Similarities for {algorithm_name}')
    plt.savefig(f'plots/similarity_{algorithm_name}')
    
    avg = np.mean(values)
    print('Mean of all similarities:', avg)
    return avg
        
if __name__ == "__main__":
    similarities = {}
    for algorithm_name, _ in clustering_algorithms.items():
        print('###################################          ', algorithm_name , '                 ###################################')

        filename = f'clustering/{algorithm_name.lower()}.csv'
        df = pd.read_csv(filename)
        sim = compute_infra_list_similarity(df, algorithm_name)
        similarities[algorithm_name.lower()] = sim
    
    print(similarities)
    keys = list(similarities.keys())
    values = list(similarities.values())
    plt.clf()
    plt.bar(keys, values)
    plt.title(f'Mean Intra List Similarities for clustering algorithms')
    plt.savefig(f'plots/overall_similarity')