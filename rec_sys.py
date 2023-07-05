import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, MeanShift, Birch
import recmetrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Cluster-based methods
n_clusters = 3

clustering_algorithms = {
    'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'),
    "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
    'DBSCAN': DBSCAN(eps=2, min_samples=5),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters, n_init="auto"),
    'MeanShift': MeanShift(bin_seeding=True),
    'GaussianMixture': GaussianMixture(
        n_components=n_clusters, covariance_type="full"
    ),
    'Birch': Birch(n_clusters=n_clusters),
}


X = np.loadtxt('PCA_X.np')

plot_num = 0
for name, algorithm in clustering_algorithms.items():
  X_std = pca_2
  plot_num += 1

  algorithm.fit(X_std)
  if hasattr(algorithm, "labels_"):
    y_pred = algorithm.labels_.astype(int)
  else:
    y_pred = algorithm.predict(X_std)

  plt.subplot(2, 3, plot_num)
  plt.subplots_adjust(hspace=1)
  plt.scatter(X_std[:, 0], X_std[:, 1], color=colors[y_pred])
  plt.title(f'PCA2: {name}')

  print(f'{name} finished')

plt.show()

# Evaluate each method using Intra-list Similarity
for method_name, method in methods.items():
    # Fit the clustering model
    clusters = method.fit_predict(movies)

    # Calculate Intra-list Similarity
    intra_list_similarity = recmetrics.intra_list_similarity(clusters, movies)

    # Print the results
    print(f"{method_name} Intra-list Similarity:", intra_list_similarity)





###########################################





import pandas as pd

# Load the dataset
df = pd.read_csv('movies_metadata.csv')

from sklearn.cluster import KMeans

# Feature engineering
# Example: One-hot encoding of movie genres
genres = df['genres'].str.split('|', expand=True)
genres = pd.get_dummies(genres.stack()).sum(level=0)

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(genres)

# Recommend movies
def recommend_movies(movie_id, num_recommendations=5):
    #todo: if movie is in a collection, recommend other movies in that collection
    movie_cluster = cluster_labels[movie_id]
    similar_movies = df[cluster_labels == movie_cluster]
    similar_movies = similar_movies[similar_movies.index != movie_id]
    similar_movies = similar_movies.sample(num_recommendations)
    return similar_movies

# Example usage
movie_id = 123
recommendations = recommend_movies(movie_id)
print(recommendations)




##########################################





# Recommend movies
def recommend_movies_cosine_sim(movie_title, top_n=5):
    # Preprocess the data

    # Feature engineering
    # Example: Combine movie genres, keywords, and cast features
    movies_df['genres'] = movies_df['genres'].fillna('[]').apply(eval)
    movies_df['keywords'] = movies_df['id'].map(keywords_df.set_index('id')['keywords'])
    movies_df['cast'] = movies_df['id'].map(credits_df.set_index('id')['cast'])

    # Create a feature matrix
    movies_df['features'] = movies_df['genres'] + movies_df['keywords'] + movies_df['cast']
    movies_df['features'] = movies_df['features'].apply(lambda x: ' '.join(x))

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['features'])

    # Compute similarity scores
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return movies_df['title'].iloc[top_movies_indices]


