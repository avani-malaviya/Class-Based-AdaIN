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

parser = argparse.ArgumentParser()
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')
parser.add_argument('--style_size', type=int, default=512, help='New (minimum) size for the style image, keeping the original size if set to 0')
parser.add_argument('--style_dir', type=str, required=True, help='Directory path to a batch of style images')
parser.add_argument('--style_mask_dir', type=str, required=True, help='Directory path to segmentation Mask of style images')
parser.add_argument('--output_file', type=str, default='style_statistics.json', help='Output file to save style statistics')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = net.vgg
vgg.eval()
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

mask_tf = mask_transform(args.style_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

style_dir = Path(args.style_dir)
style_paths = list(style_dir.glob('*'))


style_means = {}
style_stds = {}

for style_path in style_paths:
    style = style_tf(Image.open(str(style_path)))
    style_sem = get_sem_map(style_path, args.style_mask_dir)
    
    if style_sem is None:
        print(f"No segmentation mask found for {style_path}. Skipping.")
        continue
    
    style = style.to(device).unsqueeze(0)
    style_sem = style_sem.to(device).unsqueeze(0)
    style_f = vgg(style)
    
    style_means[style_path] = np.zeros((34, 512))
    style_stds[style_path] = np.zeros((34, 512))
    
    for class_id in torch.unique(style_sem):
        class_id_int = int(class_id.item())  # Convert to integer
        style_mask = F.interpolate((style_sem == class_id).float(), size=style_f.shape[2:], mode='nearest')
        style_mean, style_std = calc_weighted_mean_std(style_f, style_mask)
        
        style_means[style_path][class_id_int] = style_mean.squeeze(-1).squeeze(-1).cpu().detach().numpy()
        style_stds[style_path][class_id_int] = style_std.squeeze(-1).squeeze(-1).cpu().detach().numpy()

# Calculate average means and stds
style_mean_avg = np.mean(list(style_means.values()), axis=0)
style_std_avg = np.mean(list(style_stds.values()), axis=0)

print(style_mean_avg.shape)
print(style_std_avg.shape)

# Save the arrays
np.save('style_means.npy', style_means)
np.save('style_stds.npy', style_stds)
