import wandb
import os
import argparse
from PIL import Image

wandb.init(project='Sim2Real_AdaIN')

def log_images_recursively(folder_path, prefix=""):
    # First, check if there are any subdirectories
    subdirs = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
    
    if subdirs:
        # If there are subdirectories, recurse into them
        for subdir in subdirs:
            new_prefix = os.path.join(prefix, subdir)
            log_images_recursively(os.path.join(folder_path, subdir), new_prefix)
    else:
        # If no subdirectories, log all images in this directory
        for item in os.listdir(folder_path):
            if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, item)
                image = Image.open(image_path)
                log_path = os.path.join(prefix, item)
                wandb.log({log_path: wandb.Image(image)})

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='File path to the image directory')
args = parser.parse_args()

log_images_recursively(args.dir)
wandb.finish()