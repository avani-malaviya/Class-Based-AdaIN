import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)

def resize_image(image, target_size=512, method=Image.LANCZOS):
    return image.resize((target_size, target_size), method)

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = resize_image(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = (image * 2.0) - 1.0

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample()
    
    print(f"Shape of encoded latent: {latent.shape}")
    return latent, original_size

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

# Example usage
input_image_path = "/home/disc/a.malaviya/Desktop/coloured_cars/bremen_000040_000019_leftImg8bit.png"
latent, original_size = encode_image(input_image_path)

# Generate unsqueezed version
unsqueezed_image = decode_image(latent, original_size, keep_square=False)
unsqueezed_image.save("reconstructed_image_unsqueezed.png")

print(f"Original image size: {original_size}")



