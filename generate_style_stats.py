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

def resize_image_sdvae(image, target_size=512, method=Image.LANCZOS):
    return image.resize((target_size, target_size), method)

def encode_image_sdvae(image_path):
    image = Image.open(image_path).convert("RGB")
    image = resize_image_sdvae(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = (image * 2.0) - 1.0
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample()
    return latent

parser = argparse.ArgumentParser()
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')
parser.add_argument('--style_size', type=int, default=512, help='New (minimum) size for the style image, keeping the original size if set to 0')
parser.add_argument('--style_dir', type=str, required=True, help='Directory path to a batch of style images')
parser.add_argument('--style_mask_dir', type=str, required=True, help='Directory path to segmentation Mask of style images')
parser.add_argument('--architecture', type=str, default='encoder-decoder', help='Type of encoder-decoder architecture used')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.architecture == 'encoder-decoder':
    vgg = net.vgg
    vgg.eval()
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
elif args.architecture == 'sd-vae':
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    vae = pipe.vae
    vae = vae.to(device)
else: 
    print("invalid architecture")


mask_tf = mask_transform(args.style_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

style_dir = Path(args.style_dir)
style_paths = list(style_dir.glob('*'))

style_means = {}
style_stds = {}
style_Ns = {}

for style_path in style_paths:
    if args.architecture == 'encoder-decoder':
        style = style_tf(Image.open(str(style_path)))
        style = style.to(device).unsqueeze(0)
        style_f = vgg(style)
    elif args.architecture == 'sd-vae':
        style = style_tf(Image.open(str(style_path)))
        style_f = encode_image_sdvae(style_path)
    else: 
        print("invalid architecture")
   

    style_sem = get_sem_map(style_path, args.style_mask_dir)
    if style_sem is None:
        print(f"No segmentation mask found for {style_path}. Skipping.")
        continue
    style_sem = style_sem.to(device).unsqueeze(0)
    
    style_means[style_path] = {}
    style_stds[style_path] = {}
    style_Ns[style_path] = {}
    
    for class_id in torch.unique(style_sem):
        style_mask = F.interpolate((style_sem == class_id).float(), size=style_f.shape[2:], mode='nearest')
        style_mean, style_std, style_N = calc_weighted_mean_std(style_f, style_mask)
        class_id_float = class_id.item()
        style_means[style_path][class_id_float] = style_mean.squeeze().cpu().detach().numpy()
        style_stds[style_path][class_id_float] = style_std.squeeze().cpu().detach().numpy()
        style_Ns[style_path][class_id_float] = style_N.squeeze().cpu().detach().numpy()

serializable_style_means = {}
serializable_style_stds = {}
serializable_style_Ns = {}

for style_path in style_means:
    str_style_path = str(style_path) 
    serializable_style_means[str_style_path] = {}
    serializable_style_stds[str_style_path] = {}
    serializable_style_Ns[str_style_path] = {}
    for class_id in style_means[style_path]:
        serializable_style_means[str_style_path][class_id] = style_means[style_path][class_id].tolist()
        serializable_style_stds[str_style_path][class_id] = style_stds[style_path][class_id].tolist()
        serializable_style_Ns[str_style_path][class_id] = style_Ns[style_path][class_id].tolist()


with open('style_means.json', 'w') as f:
    json.dump(serializable_style_means, f)

with open('style_stds.json', 'w') as f:
    json.dump(serializable_style_stds, f)


if args.architecture == 'encoder-decoder':
    C = 512
elif args.architecture == 'sd-vae':
    C = 4
else: 
    print("invalid architecture")


accumulated_means = {}
total_N = {}

eps = 1e-5

for style_path, class_means in style_means.items():
    for class_id, mean_value in class_means.items():
        if class_id not in accumulated_means:
            accumulated_means[class_id] = 0.0 * np.ones(C)
            total_N[class_id] = 0
        N = style_Ns[style_path][class_id]
        accumulated_means[class_id] += N * mean_value
        total_N[class_id] += N

for class_id in accumulated_means:
    total_N[class_id] += eps
    accumulated_means[class_id] /= total_N[class_id]

with open("multi_ref_means.txt", "wb") as myFile:
    pickle.dump(accumulated_means, myFile)


accumulated_vars = {}
total_N = {}

for style_path, class_stds in style_stds.items():
    for class_id, std_value in class_stds.items():
        if class_id not in accumulated_vars:
            accumulated_vars[class_id] = 0.0 * np.ones(C)
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

with open("multi_ref_stds.txt", "wb") as myFile:
    pickle.dump(accumulated_stds, myFile)





    


