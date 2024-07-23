import argparse
from pathlib import Path
import re

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import net


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



def resize_image_sdvae(image, target_size=512, method=Image.LANCZOS):
    return image.resize((target_size, target_size), method)

def encode_image_sdvae(image_path):
    image = Image.open(image_path).convert("RGB")
    content_size = image.size
    image = resize_image_sdvae(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = (image * 2.0) - 1.0

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample()
    
    return latent, content_size


def decode_image_sdvae(latent, original_size, keep_square=False):
    with torch.no_grad():
        image = vae.decode(latent).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    if not keep_square:
        image = image.resize(original_size, Image.LANCZOS)
    
    return image


def style_transfer(adain, content_f, content_sem, style_f = None, style_sem = None, style_means = None, style_stds = None, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    if ((style_means is not None) and (style_stds is not None)):
        feat = adain(content_f, content_sem, style_means, style_stds)
    elif ((style_f is not None) and (style_sem is not None)):
        feat = adain(content_f, style_f, content_sem, style_sem)
    else: 
        print("Incomplete Style Data")
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
parser.add_argument('--style_files', type=str,
                    help='Comma separated paths to .txt files containing class based style means and stds respectively')
parser.add_argument('--content_mask_dir',type=str, required=True, 
                    help='Directory path to segmantation Mask of content images')
parser.add_argument('--style_mask_dir',type=str, 
                    help='Directory path to segmantation Mask of style images')
parser.add_argument('--architecture', type=str, default='encoder-decoder', help='Type of encoder-decoder architecture used')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--adain_method', type=str, required=True)


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
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')


args = parser.parse_args()

if (args.adain_method=="saved_stats"):
    from function import adaptive_instance_normalization_saved_stats as adain
elif (args.adain_method=="with_segmentation"):
    from function import adaptive_instance_normalization_by_segmentation as adain
else:
    from function import adaptive_instance_normalization as adain


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


# Either --style or --styleDir or --style_files should be given.
assert (args.style or args.style_dir or args.style_files) 
if args.style:
    style_paths = [Path(args.style)]
elif args.style_dir:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
else:
    style_means, style_stds = args.style_files.split(',')


device = "cuda" if torch.cuda.is_available() else "cpu"

if args.architecture == 'encoder-decoder':
    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    decoder.to(device)
elif args.architecture == 'sd-vae':
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    vae = pipe.vae
    vae = vae.to(device)
else: 
    print("invalid architecture")


content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

content_mask_tf = mask_transform(args.content_size, args.crop)
style_mask_tf = mask_transform(args.style_size, args.crop)

for content_path in content_paths:
    
    if args.adain_method=="saved_stats":
        content = content_tf(Image.open(str(content_path)))
        content_sem = get_sem_map(content_path,args.content_mask_dir)
        content = content.to(device).unsqueeze(0)
        content_sem = content_sem.to(device).unsqueeze(0)

        if args.architecture == 'encoder-decoder':
            content_f = vgg(content)
        elif args.architecture == 'sd-vae':
            content_f, content_size = encode_image_sdvae(content_path)

        with torch.no_grad():
            adain_feat = style_transfer(adain, content_f, content_sem, style_means=style_means, style_stds=style_stds, alpha=args.alpha)
        
        if args.architecture == 'encoder-decoder':
            output = decoder(adain_feat)
            output = output.cpu()
        elif args.architecture == 'sd-vae':
            output = decode_image_sdvae(adain_feat, content_size)

        output_name = output_dir / '{:s}_stylized{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            
            content_sem = get_sem_map(content_path,args.content_mask_dir)
            style_sem = get_sem_map(style_path,args.style_mask_dir)
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            content_sem = content_sem.to(device).unsqueeze(0)
            style_sem = style_sem.to(device).unsqueeze(0)

            if args.architecture == 'encoder-decoder':
                content_f = vgg(content)
            elif args.architecture == 'sd-vae':
                content_f, content_size = encode_image_sdvae(content_path)

            with torch.no_grad():
                adain_feat = style_transfer(adain, vgg, decoder, content, content_sem, style=style, style_sem=style_sem, alpha=args.alpha)
            
            if args.architecture == 'encoder-decoder':
                output = decoder(adain_feat)
                output = output.cpu()
            elif args.architecture == 'sd-vae':
                output = decode_image_sdvae(adain_feat, content_size)

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
