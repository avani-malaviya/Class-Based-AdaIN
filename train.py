import argparse
from pathlib import Path

import wandb
import re
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, mask_root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.mask_root = mask_root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        self.mask_suffix = '_gtFine_labelIds.png'

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)

        # Extract the base name and replace the suffix
        match = re.match(r'(.*)\.png$', path.name)
        if match:
            base_name = match.group(1)
            mask_path = str(Path(self.mask_root) / (base_name + self.mask_suffix))
            if Path(mask_path).exists():
                mask = Image.open(mask_path)
                mask = self.transform(mask)
                mask = np.array(mask)
            else:
                mask = None
        else:
            mask = None

        return img, mask

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth',
                    help='Path to the pretrained decoder model')
parser.add_argument('--with_segmentation', type=bool, required=True)

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=200000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=0.0)
parser.add_argument('--content_weight', type=float, default=10.0)
#parser.add_argument('--reg_weight', type=float, default=1000.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--content_mask_dir',type=str, required=True, 
                    help='Directory path to segmantation Mask of content images')
parser.add_argument('--style_mask_dir',type=str, required=True, 
                    help='Directory path to segmantation Mask of style images')
args = parser.parse_args()

if args.with_segmentation:
    from function import adaptive_instance_normalization_by_segmentation as adain
else:
    from function import adaptive_instance_normalization as adain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

wandb.init(project="Sim2Real_AdaIN", config={
    "content_dir": args.content_dir,
    "style_dir": args.style_dir,
    "vgg": args.vgg,
    "save_dir": args.save_dir,
    "log_dir": args.log_dir,
    "lr": args.lr,
    "lr_decay": args.lr_decay,
    "max_iter": args.max_iter,
    "batch_size": args.batch_size,
    "style_weight": args.style_weight,
    "content_weight": args.content_weight,
#    "regularization_weight": args.reg_weight,
    "n_threads": args.n_threads,
    "save_model_interval": args.save_model_interval
})

decoder = net.decoder
decoder.load_state_dict(torch.load(args.decoder))
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(adain, vgg, decoder)
network.train()
network.to(device)

wandb.watch(network, log="all")

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, args.content_mask_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, args.style_mask_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images, content_mask = next(content_iter)
    content_images = content_images.to(device)
    content_mask = content_mask.to(device)
    style_images, style_mask = next(style_iter)
    style_images = style_images.to(device)
    style_mask = style_mask.to(device)
    loss_c, loss_s = network(content_images, style_images, content_mask, style_mask)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
#    loss_m = args.reg_weight * loss_m
    loss = loss_c + loss_s # + loss_m

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({"loss_content": loss_c.item(), "loss_style": loss_s.item()}, step=i)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'decoder_iter_{:d}.pth.tar'.format(i + 1))
