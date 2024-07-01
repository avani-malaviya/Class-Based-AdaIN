import os
import sys
import argparse
import cv2

def combine_images(dir_content, dir_style, dir_stylized, output_dir):
    # Loop through all files in the stylized directory
    for filename in os.listdir(dir_stylized):
        path = os.path.join(dir_stylized, filename)
        if os.path.isdir(path):
            # skip directories
            continue
        # Extract the base names
        imgname, ext = os.path.splitext(filename)
        base_content, base_style = imgname.split("_stylized_")

        # Load the images
        img_stylized = cv2.imread(os.path.join(dir_stylized, filename))
        img_content = cv2.imread(os.path.join(dir_content, base_content + ".png"))
        img_style = cv2.imread(os.path.join(dir_style, base_style + ".png"))

        # Resize the content and style images to match the stylized image
        img_content = cv2.resize(img_content, (img_stylized.shape[1], img_stylized.shape[0]))
        img_style = cv2.resize(img_style, (img_stylized.shape[1], img_stylized.shape[0]))

        # Concatenate the images horizontally
        img_combined = cv2.hconcat([img_content, img_style, img_stylized])

        # Save the combined image
        combined_filename = os.path.join(output_dir, f"{base_content}_{base_style}_combined.jpg")
        cv2.imwrite(combined_filename, img_combined)
        print(f"Combined image saved as {combined_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine images from different directories')
    parser.add_argument('--dir_content', type=str, required=True, help='Path to content directory')
    parser.add_argument('--dir_style', type=str, required=True, help='Path to style directory')
    parser.add_argument('--dir_stylized', type=str, required=True, help='Path to directory containing stylized images')
    parser.add_argument('--dir_output', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    combine_images(args.dir_content, args.dir_style, args.dir_stylized, args.dir_output)
