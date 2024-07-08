import argparse
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

import net
from function import coral


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

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def mask_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size, interpolation=Image.NEAREST))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def visualize_feature_maps(content_f, output_f, output_name):
    # Ensure the tensors are on CPU and detached from the computation graph
    content_f = content_f.cpu().detach()
    output_f = output_f.cpu().detach()

    # Normalize feature maps
    content_f = (content_f - content_f.min()) / (content_f.max() - content_f.min())
    output_f = (output_f - output_f.min()) / (output_f.max() - output_f.min())
    
    # Create a grid of feature maps
    num_features = content_f.size(1)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # Create a large figure to hold all feature maps
    fig, axs = plt.subplots(2, grid_size, figsize=(grid_size*2, 4), dpi=750)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(num_features):
        row = i // grid_size
        col = i % grid_size
        
        # Content feature map
        axs[0, col].imshow(content_f[0, i].numpy(), cmap='gray')
        axs[0, col].axis('off')
        
        # Output feature map
        axs[1, col].imshow(output_f[0, i].numpy(), cmap='gray')
        axs[1, col].axis('off')

    # Remove any unused subplots
    for i in range(num_features, grid_size):
        axs[0, i].axis('off')
        axs[1, i].axis('off')

    # Save the figure
    plt.savefig(str(output_name.with_stem(output_name.stem + '_feature_maps')), 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def process_styles(style_paths, args, vgg, device):
    all_style_f = []
    all_style_masks = {}
    all_classes = set()

    for style_path in style_paths:
        style = style_tf(Image.open(str(style_path)))
        style_sem = get_sem_map(style_path, args.style_mask_dir)
        
        if style_sem is None:
            print(f"No segmentation mask found for {style_path}. Skipping.")
            continue
        
        style = style.to(device).unsqueeze(0)
        style_sem = style_sem.to(device).unsqueeze(0)
        style_f = vgg(style)
        all_style_f.append(style_f)
        
        classes = torch.unique(style_sem)
        all_classes.update(classes.cpu().numpy())
        
        for class_id in classes:
            if class_id not in all_style_masks:
                all_style_masks[class_id] = []
            style_mask = F.interpolate((style_sem == class_id).float(), size=style_f.shape[2:], mode='nearest')
            all_style_masks[class_id].append(style_mask)

    # Stack all style features
    stacked_style_f = torch.cat(all_style_f, dim=0)

    # Create stacked masks for all classes, filling with zeros where necessary
    stacked_style_masks = {}
    for class_id in all_classes:
        if class_id in all_style_masks:
            masks = all_style_masks[class_id]
            while len(masks) < len(style_paths):
                masks.append(torch.zeros_like(masks[0]))
            stacked_style_masks[class_id] = torch.cat(masks, dim=0)
        else:
            stacked_style_masks[class_id] = torch.zeros((len(style_paths), 1, *style_f.shape[2:]), device=device)

    return stacked_style_f, stacked_style_masks


def style_transfer(adain, vgg, decoder, content, style, content_sem, style_sem, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adain(content_f, style_f,content_sem,style_sem)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adain(content_f, style_f,content_sem,style_sem)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat), content_f


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--with_segmentation', type=str, required=True)

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('--content_mask_dir',type=str, required=True, 
                    help='Directory path to segmantation Mask of content images')
parser.add_argument('--style_mask_dir',type=str, required=True, 
                    help='Directory path to segmantation Mask of style images')

args = parser.parse_args()


if (args.with_segmentation=="True"):
    from function import adaptive_instance_normalization_by_segmentation as adain
elif (args.with_segmentation=="precomputed"):
    from function import adaptive_instance_normalization_precalculated as adain
else:
    from function import adaptive_instance_normalization as adain

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

content_mask_tf = mask_transform(args.content_size, args.crop)
style_mask_tf = mask_transform(args.style_size, args.crop)


for content_path in content_paths:

    if (args.with_segmentation=="precomputed"):
        content = content_tf(Image.open(str(content_path)))
        content_sem = get_sem_map(content_path,args.content_mask_dir)
        content = content.to(device).unsqueeze(0)
        content_sem = content_sem.to(device).unsqueeze(0)

        stacked_style_f, stacked_style_masks = process_styles(style_paths, args, vgg, device)

        with torch.no_grad():
            output, content_f = style_transfer(adain, vgg, decoder, content, stacked_style_f, content_sem, stacked_style_masks,
                                    args.alpha)
            output_f = vgg(output)
        output = output.cpu()

        output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            content_path.stem, style_path.stem, args.save_ext)
        save_image(output, str(output_name))
    
    elif do_interpolation:
        style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
        style_sem = torch.stack([get_sem_map(Path(p),args.style_mask_dir).to(device) for p in style_paths])
        content_image = Image.open(str(content_path))
        style_height, style_width = style.shape[2], style.shape[3]
        content_image = Resize((style_height, style_width))(content_image)
        content = content_tf(content_image).unsqueeze(0)
        content = content.expand_as(style)
        content_sem = get_sem_map(content_path,args.content_mask_dir).to(device).unsqueeze(0)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output, content_f = style_transfer(adain, vgg, decoder, content, style, content_sem, style_sem,
                                    args.alpha, interpolation_weights)
            output_f = vgg(output)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))
        #visualize_feature_maps(content_f, output_f, output_name)

    else:
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            
            content_sem = get_sem_map(content_path,args.content_mask_dir)
            style_sem = get_sem_map(style_path,args.style_mask_dir)
            
            if args.preserve_color:
                style = coral(style, content)
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            content_sem = content_sem.to(device).unsqueeze(0)
            style_sem = style_sem.to(device).unsqueeze(0)

            with torch.no_grad():
                output, content_f = style_transfer(adain, vgg, decoder, content, style, content_sem, style_sem,
                                        args.alpha)
                output_f = vgg(output)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
            #visualize_feature_maps(content_f, output_f, output_name)
