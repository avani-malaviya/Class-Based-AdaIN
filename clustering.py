import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
from pathlib import Path
import torch
import shutil
import argparse

parser = argparse.ArgumentParser(description='Cluster and visualize style images.')
parser.add_argument('--class_id', type=float, default=0.09019608050584793, help='Class ID to process')
parser.add_argument('--style_means_path', type=str, default='style_means.json', help='Path to style_means.json')
parser.add_argument('--style_stds_path', type=str, default='style_stds.json', help='Path to style_stds.json')
parser.add_argument('--mask_dir', type=str, default='input/style/cityscapes/labels/', help='Directory containing mask files')
parser.add_argument('--output_dir', type=str, default='input/style/cityscapes/images/sky_clusters/', help='Output directory for clustered images')
parser.add_argument('--visualization_dir', type=str, default='output/sim2real/cluster_visualizations', help='Output directory for cluster visualizations')
parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')
parser.add_argument('--visualize', action='store_true', help='Flag to generate cluster visualizations')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
with open(args.style_means_path, 'r') as f:
    style_means = json.load(f)
with open(args.style_stds_path, 'r') as f:
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

def prepare_data(means_dict, class_id):
    data = []
    style_paths = []
    for style_path in means_dict:
        if str(class_id) in means_dict[style_path]:
            features = means_dict[style_path][str(class_id)]
            data.append(features)
            style_paths.append(style_path)
    return np.array(data), style_paths

def cluster_data(data, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    similarity_matrix = cosine_similarity(X_scaled)
    distance_matrix = 1 - similarity_matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    return cluster_labels, kmeans.cluster_centers_

def visualize_clusters(style_paths, cluster_labels, class_id, n_clusters, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cluster_images = {i: [] for i in range(n_clusters)}
    for path, label in zip(style_paths, cluster_labels):
        cluster_images[label].append(path)
    for cluster in range(n_clusters):
        paths = cluster_images[cluster]
        n_images = len(paths)
        if n_images == 0:
            print(f"No images in cluster {cluster} for class {class_id}")
            continue
        grid_size = max(1, int(np.ceil(np.sqrt(n_images))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f"Class {class_id} - Cluster {cluster}")
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and axes.ndim == 1:
            axes = axes.reshape(1, -1)
        for i, path in enumerate(paths):
            style_sem = get_sem_map(Path(path), args.mask_dir)
            style_sem = style_sem.squeeze().numpy()
            binary_mask = np.array(style_sem == float(class_id), dtype=np.uint8) 
            img = Image.open(path)
            binary_mask = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
            img_array = np.array(img)
            masked_img_array = img_array * binary_mask
            img = Image.fromarray(masked_img_array)
            ax = axes[i // grid_size, i % grid_size]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(path), fontsize=8)
        for i in range(n_images, grid_size * grid_size):
            fig.delaxes(axes.flatten()[i])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_{class_id}_cluster_{cluster}.png'))
        plt.close()

def save_clustered_images(style_paths, cluster_labels, class_id, output_dir):
    class_output_dir = os.path.join(output_dir, f'class_{class_id}')
    os.makedirs(class_output_dir, exist_ok=True)
    for i, (cluster_label, style_path) in enumerate(zip(cluster_labels, style_paths)):
        cluster_dir = os.path.join(class_output_dir, f'cluster_{cluster_label}')
        os.makedirs(cluster_dir, exist_ok=True)
        if not os.path.exists(style_path):
            print(f"File not found: {style_path}")
            continue
        try:
            output_filename = os.path.join(cluster_dir, os.path.basename(style_path))
            shutil.copy2(style_path, output_filename)
            print(f"Copied image {i+1}/{len(style_paths)}: {output_filename}")
        except Exception as e:
            print(f"Error processing {style_path}: {str(e)}")
    print(f"Images for class {class_id} copied to {class_output_dir}")

# Main execution
data, style_paths = prepare_data(style_means, args.class_id)
cluster_labels, cluster_centers = cluster_data(data, args.n_clusters)

if args.visualize:
    if args.visualization_dir:
        visualize_clusters(style_paths, cluster_labels, args.class_id, args.n_clusters, args.visualization_dir)
        print(f"Visualizations saved in {args.visualization_dir}")
    else:
        print("Visualization directory not provided. Skipping visualization.")

save_clustered_images(style_paths, cluster_labels, args.class_id, args.output_dir)

print(f"Class {args.class_id}:")
print(f"  Number of samples: {len(cluster_labels)}")
print("  Cluster sizes:")
for i in range(args.n_clusters):
    cluster_size = np.sum(cluster_labels == i)
    print(f"    Cluster {i}: {cluster_size}")