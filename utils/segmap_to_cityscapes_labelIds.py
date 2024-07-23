import os
import numpy as np
import cv2

# Define the mapping from RGB to label ID for all Cityscapes labels
rgb_to_label = {
    (0, 0, 0): 0,         # unlabeled
    (0, 0, 0): 1,         # ego vehicle
    (0, 0, 0): 2,         # rectification border
    (0, 0, 0): 3,         # out of roi
    (0, 0, 0): 4,         # static
    (111, 74, 0): 5,      # dynamic
    (81, 0, 81): 6,       # ground
    (128, 64, 128): 7,    # road
    (244, 35, 232): 8,    # sidewalk
    (250, 170, 160): 9,   # parking
    (230, 150, 140): 10,  # rail track
    (70, 70, 70): 11,     # building
    (102, 102, 156): 12,  # wall
    (190, 153, 153): 13,  # fence
    (180, 165, 180): 14,  # guard rail
    (150, 100, 100): 15,  # bridge
    (150, 120, 90): 16,   # tunnel
    (153, 153, 153): 17,  # pole
    (153, 153, 153): 18,  # polegroup
    (250, 170, 30): 19,   # traffic light
    (220, 220, 0): 20,    # traffic sign
    (107, 142, 35): 21,   # vegetation
    (152, 251, 152): 22,  # terrain
    (70, 130, 180): 23,   # sky
    (220, 20, 60): 24,    # person
    (255, 0, 0): 25,      # rider
    (0, 0, 142): 26,      # car
    (0, 0, 70): 27,       # truck
    (0, 60, 100): 28,     # bus
    (0, 0, 90): 29,       # caravan
    (0, 0, 110): 30,      # trailer
    (0, 80, 100): 31,     # train
    (0, 0, 230): 32,      # motorcycle
    (119, 11, 32): 33,    # bicycle
    (0, 0, 142): -1,      # license plate
}

def rgb_to_label_id(rgb_image):
    """Convert RGB image to label ID image."""
    label_id_data = np.zeros(rgb_image.shape[:2], dtype=np.int16)
    
    for rgb, label_id in rgb_to_label.items():
        mask = np.all(rgb_image == rgb, axis=-1)
        label_id_data[mask] = label_id
    
    return label_id_data

def process_directory(input_dir, output_dir):
    """Process all PNG files in the input directory and save results in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Read image using OpenCV
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError("Failed to read image")
                
                # OpenCV reads images in BGR, so convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Convert to label IDs
                label_id_img = rgb_to_label_id(img_rgb)
                
                # Save the result
                cv2.imwrite(output_path, label_id_img.astype(np.uint8))
                
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


# Usage
input_directory = 'input/content/Cityscapes/labels_original/'
output_directory = 'input/content/Cityscapes/labels/'
process_directory(input_directory, output_directory)

input_directory = 'input/style/GTA/labels_original/'
output_directory = 'input/style/GTA/labels/'
process_directory(input_directory, output_directory)
