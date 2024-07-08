import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the data
with open('style_means.json', 'r') as f:
    style_means = json.load(f)
with open('style_stds.json', 'r') as f:
    style_stds = json.load(f)

# Prepare the data for clustering
def prepare_combined_data(means_dict, stds_dict):
    combined_data = {}
    for style_path in means_dict:
        for class_id in means_dict[style_path]:
            if class_id not in combined_data:
                combined_data[class_id] = []
            # Concatenate mean and std features
            combined_features = means_dict[style_path][class_id] + stds_dict[style_path][class_id]
            combined_data[class_id].append((style_path, combined_features))
    return combined_data

combined_data = prepare_combined_data(style_means, style_stds)

# Perform clustering for each class
n_clusters = 5  # You can adjust this number

def cluster_data(data, max_clusters):
    clustered_data = {}
    for class_id, features_list in data.items():
        style_paths, features = zip(*features_list)
        
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
            'cluster_centers': kmeans.cluster_centers_,
            'style_paths': style_paths
        }
    return clustered_data

clustered_combined = cluster_data(combined_data, n_clusters)

# Visualization function (same as before)
def visualize_clusters(clustered_data, class_id, n_clusters, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get cluster labels and style paths for this class
    cluster_labels = clustered_data[class_id]['cluster_labels']
    style_paths = clustered_data[class_id]['style_paths']

    # Group style paths by cluster
    cluster_images = {i: [] for i in range(n_clusters)}
    for path, label in zip(style_paths, cluster_labels):
        cluster_images[label].append(path)

    # Visualize images for each cluster
    for cluster in range(n_clusters):
        paths = cluster_images[cluster]
        n_images = len(paths)
        
        if n_images == 0:
            print(f"No images in cluster {cluster} for class {class_id}")
            continue

        # Determine grid size
        grid_size = max(1, int(np.ceil(np.sqrt(n_images))))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f"Class {class_id} - Cluster {cluster}")

        # Make sure axes is always 2D
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i, path in enumerate(paths):
            img = Image.open(path)
            ax = axes[i // grid_size, i % grid_size]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(path), fontsize=8)

        # Remove unused subplots
        for i in range(n_images, grid_size * grid_size):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_{class_id}_cluster_{cluster}.png'))
        plt.close()

# Visualize combined clusters
output_dir = 'output/sim2real/cluster_visualizations'
for class_id in clustered_combined:
    visualize_clusters(clustered_combined, class_id, n_clusters, output_dir)

print(f"Visualizations saved in {output_dir}")

# Analyze the results
for class_id in clustered_combined:
    print(f"Class {class_id}:")
    print(f"  Number of samples: {len(clustered_combined[class_id]['cluster_labels'])}")
    print("  Cluster sizes:")
    for i in range(n_clusters):
        cluster_size = np.sum(clustered_combined[class_id]['cluster_labels'] == i)
        print(f"    Cluster {i}: {cluster_size}")
    print()