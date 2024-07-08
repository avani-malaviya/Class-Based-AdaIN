import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the data (same as before)
with open('style_means.json', 'r') as f:
    style_means = json.load(f)
with open('style_stds.json', 'r') as f:
    style_stds = json.load(f)

# 2. Prepare the data for clustering
def prepare_data(data_dict):
    prepared_data = {}
    for style_path, class_data in data_dict.items():
        for class_id, features in class_data.items():
            if class_id not in prepared_data:
                prepared_data[class_id] = []
            prepared_data[class_id].append(features)
    return prepared_data

means_data = prepare_data(style_means)
stds_data = prepare_data(style_stds)

# 3. Perform clustering for each class
n_clusters = 5  # You can adjust this number

def cluster_data(data, max_clusters):
    clustered_data = {}
    for class_id, features in data.items():
        # Convert list of lists to numpy array
        X = np.array(features)
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(X_scaled)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        n_clusters = min(max_clusters, X_scaled.shape[0])
        
        # Perform K-means clustering with precomputed distances
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        clustered_data[class_id] = {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_
        }
    return clustered_data

clustered_means = cluster_data(means_data, n_clusters)
clustered_stds = cluster_data(stds_data, n_clusters)

# 4. Analyze the results (same as before)
for class_id in clustered_means:
    print(f"Class {class_id}:")
    print(f"  Number of samples: {len(clustered_means[class_id]['cluster_labels'])}")
    print("  Cluster sizes:")
    for i in range(n_clusters):
        cluster_size = np.sum(clustered_means[class_id]['cluster_labels'] == i)
        print(f"    Cluster {i}: {cluster_size}")
    print()