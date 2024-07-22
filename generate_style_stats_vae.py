import argparse
from pathlib import Path
import re
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import net
from function import calc_weighted_mean_std
import numpy as np
import pickle
import json
from diffusers import StableDiffusionPipeline



def mask_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size, interpolation=Image.NEAREST))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_sem_map(img_path, mask_dir):
    match = re.match(r'(.*)\.png$', img_path.name)
    if match:
        base_name = match.group(1)
        mask_path = Path(mask_dir) / (base_name + '_gtFine_labelIds.png')
        if mask_path.exists():
            return mask_tf(Image.open(str(mask_path)).convert('L'))
    return None

def resize_image(image, target_size=512, method=Image.LANCZOS):
    return image.resize((target_size, target_size), method)

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = (image * 2.0) - 1.0

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample()
    
    return latent


def decode_image(latent, original_size, keep_square=False):
    with torch.no_grad():
        image = vae.decode(latent).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    if not keep_square:
        image = image.resize(original_size, Image.LANCZOS)
    
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')
parser.add_argument('--style_size', type=int, default=512, help='New (minimum) size for the style image, keeping the original size if set to 0')
parser.add_argument('--style_dir', type=str, required=True, help='Directory path to a batch of style images')
parser.add_argument('--style_mask_dir', type=str, required=True, help='Directory path to segmentation Mask of style images')
args = parser.parse_args()



model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
vae = pipe.vae

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)

mask_tf = mask_transform(args.style_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

style_dir = Path(args.style_dir)
style_paths = list(style_dir.glob('*'))

style_means = {}
style_stds = {}
style_Ns = {}

for style_path in style_paths:
    style = style_tf(Image.open(str(style_path)))
    style_sem = get_sem_map(style_path, args.style_mask_dir)
    
    if style_sem is None:
        print(f"No segmentation mask found for {style_path}. Skipping.")
        continue
    
    style_f = encode_image(style_path)

    style_sem_resized = F.interpolate(style_sem.unsqueeze(0).float(), 
                                  size=style_f.shape[2:], 
                                  mode='nearest').squeeze(0).squeeze(0)
    
    style_means[style_path] = {}
    style_stds[style_path] = {}
    style_Ns[style_path] = {}
    
    for class_id in torch.unique(style_sem_resized):
        style_mask = (style_sem_resized == class_id).float().unsqueeze(0).to(device)
        style_mean, style_std, style_N = calc_weighted_mean_std(style_f, style_mask)
        class_id_float = class_id.item()
        style_means[style_path][class_id_float] = style_mean.squeeze().cpu().detach().numpy()
        style_stds[style_path][class_id_float] = style_std.squeeze().cpu().detach().numpy()
        style_Ns[style_path][class_id_float] = style_N.squeeze().cpu().detach().numpy()

serializable_style_means = {}
serializable_style_stds = {}
serializable_style_Ns = {}

for style_path in style_means:
    str_style_path = str(style_path)  # Convert PosixPath to string
    serializable_style_means[str_style_path] = {}
    serializable_style_stds[str_style_path] = {}
    serializable_style_Ns[str_style_path] = {}
    for class_id in style_means[style_path]:
        serializable_style_means[str_style_path][class_id] = style_means[style_path][class_id].tolist()
        serializable_style_stds[str_style_path][class_id] = style_stds[style_path][class_id].tolist()
        serializable_style_Ns[str_style_path][class_id] = style_Ns[style_path][class_id].tolist()

# Save the dictionaries as JSON files
with open('style_means.json', 'w') as f:
    json.dump(serializable_style_means, f)

with open('style_stds.json', 'w') as f:
    json.dump(serializable_style_stds, f)


accumulated_means = {}
total_N = {}

eps = 1e-5

for style_path, class_means in style_means.items():
    for class_id, mean_value in class_means.items():
        if class_id not in accumulated_means:
            accumulated_means[class_id] = 0.0 * np.ones(4)
            total_N[class_id] = 0
        N = style_Ns[style_path][class_id]
        accumulated_means[class_id] += N * mean_value
        total_N[class_id] += N

for class_id in accumulated_means:
    total_N[class_id] += eps
    accumulated_means[class_id] /= total_N[class_id]

with open("mean_means.txt", "wb") as myFile:
    pickle.dump(accumulated_means, myFile)

accumulated_vars = {}
total_N = {}

for style_path, class_stds in style_stds.items():
    for class_id, std_value in class_stds.items():
        if class_id not in accumulated_vars:
            accumulated_vars[class_id] = 0.0 * np.ones(4)
            total_N[class_id] = 0
        N = style_Ns[style_path][class_id]
        mean_value = style_means[style_path][class_id]
        accumulated_vars[class_id] += (N - 1) * std_value**2 + N * mean_value**2
        total_N[class_id] += N

accumulated_stds = {}
for class_id in accumulated_vars:
    total_N[class_id] += eps
    accumulated_vars[class_id] -= total_N[class_id] * accumulated_means[class_id]**2
    accumulated_vars[class_id] /= (total_N[class_id] - 1)
    accumulated_stds[class_id] = np.sqrt(accumulated_vars[class_id])

with open("mean_stds.txt", "wb") as myFile:
    pickle.dump(accumulated_stds, myFile)





    


