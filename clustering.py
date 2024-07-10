import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
with open('style_means.json', 'r') as f:
    style_means = json.load(f)
with open('style_stds.json', 'r') as f:
    style_stds = json.load(f)

def mask_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size, interpolation=Image.NEAREST))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

content_mask_tf = mask_transform(0, None)

def get_sem_map(img_path, mask_dir):
    # Process content mask
    match = re.match(r'(.*)\.png$', img_path.name)
    if match:
        base_name = match.group(1)
        mask_path = str(Path(mask_dir) / (base_name + '_gtFine_labelIds.png'))
        if Path(mask_path).exists():
            sem_map = content_mask_tf(Image.open(str(mask_path)).convert('L'))
        else:
            sem_map = None
    else:
        sem_map = None

    return sem_map

# Prepare the data for clustering
def prepare_combined_data(means_dict, stds_dict):
    combined_data = {}
    for style_path in means_dict:
        for class_id in means_dict[style_path]:
            if class_id not in combined_data:
                combined_data[class_id] = []
            # Concatenate mean and std features
            combined_features = stds_dict[style_path][class_id] #+ stds_dict[style_path][class_id]
            combined_data[class_id].append((style_path, combined_features))
    return combined_data

combined_data = prepare_combined_data(style_means, style_stds)

# Perform clustering for each class
n_clusters = 5  # You can adjust this number

from sklearn.metrics.pairwise import cosine_distances

def cluster_data(data, eps=0.5, min_samples=5):
    clustered_data = {}
    for class_id, features_list in data.items():
        style_paths, features = zip(*features_list)
        
        # Convert list of lists to numpy array
        X = np.array(features)
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute cosine distances directly
        distance_matrix = cosine_distances(X_scaled)
        
        n_samples = X_scaled.shape[0]
        
        if n_samples < min_samples:
            # Handle the case where there are too few samples
            clustered_data[class_id] = {
                'cluster_labels': np.zeros(n_samples, dtype=int),
                'style_paths': style_paths
            }
        else:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            cluster_labels = dbscan.fit_predict(distance_matrix)
            
            clustered_data[class_id] = {
                'cluster_labels': cluster_labels,
                'style_paths': style_paths
            }
    
    return clustered_data


# Use the modified cluster_data function
clustered_combined = cluster_data(combined_data, eps=0.5, min_samples=5)

# Modify the visualization function for DBSCAN results
def visualize_clusters(clustered_data, class_id, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get cluster labels and style paths for this class
    cluster_labels = clustered_data[class_id]['cluster_labels']
    style_paths = clustered_data[class_id]['style_paths']

    # Get unique cluster labels
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    # Group style paths by cluster
    cluster_images = {label: [] for label in unique_labels}
    for path, label in zip(style_paths, cluster_labels):
        cluster_images[label].append(path)

    # Visualize images for each cluster
    for cluster in unique_labels:
        paths = cluster_images[cluster]
        n_images = len(paths)
        
        if n_images == 0:
            print(f"No images in cluster {cluster} for class {class_id}")
            continue

        # Determine grid size
        grid_size = max(1, int(np.ceil(np.sqrt(n_images))))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        fig.suptitle(f"Class {class_id} - {cluster_name}")

        # Make sure axes is always 2D
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i, path in enumerate(paths):
            style_sem = get_sem_map(Path(path), 'input/style/cityscapes/labels/')
            style_sem = style_sem.squeeze().numpy()
            binary_mask = np.array(style_sem == float(class_id), dtype=np.uint8) 
            img = Image.open(path)
            binary_mask = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)  # Adjust the mask for 3 channels
            img_array = np.array(img)
            masked_img_array = img_array * binary_mask
            img = Image.fromarray(masked_img_array)
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

# Visualize DBSCAN clusters
output_dir = 'output/sim2real/dbscan_cluster_visualizations'
for class_id in clustered_combined:
    visualize_clusters(clustered_combined, class_id, output_dir)

print(f"Visualizations saved in {output_dir}")

# Analyze the DBSCAN results
for class_id in clustered_combined:
    print(f"Class {class_id}:")
    cluster_labels = clustered_combined[class_id]['cluster_labels']
    n_samples = len(cluster_labels)
    print(f"  Number of samples: {n_samples}")
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    print("  Cluster sizes:")
    for label in unique_labels:
        if label != -1:
            cluster_size = np.sum(cluster_labels == label)
            print(f"    Cluster {label}: Size = {cluster_size}")
    print()