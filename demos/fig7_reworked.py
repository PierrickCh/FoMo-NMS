import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, resize


from train import *
import fourier_modules as FM
from torchvision.utils import save_image
from PIL import Image

# Parameters
USE_RANDOM_NOISE_MASK = False  # Set to False to load from a file instead
MASK_FILE_PATH = "demos/mask_05.jpg"  # Path to the mask file

dir_fig = 'demos/images/figures/'
save_dir = "demos/images/figures/complex_masks_inpainting"

method = "fomonms_nms_size_3"

img_idx = 8 # reference image to use
number_of_image_batch = 1 # number of images to create (batches)
ratio = (2, 2) # upscale ratio

# Functions
def load_model(image_idx)-> DataLoader | Model:
    # find run
    for d in os.listdir("./runs"):
        if str(image_idx) == d.split("_")[1] and "fomonms_size_3_EMN_1_CA_1" in d:
            name = d
            break
    
    # load args and model
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args = load_args(os.path.join("./runs", name, "args.json"), args)
    FM.CA = args.CA
    FM.EMN = args.EMN
    dset = DS_rot(
        args.dataset_path,
        image_size=(args.img_size, args.img_size),
        min_mask_shape=(64, 64),
        max_mask_shape=(200, 200),
    )
    
    loader = DataLoader(
        dset,
        batch_size=number_of_image_batch,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    M = Model(
        nc_im=3,
        nc_start=args.dim,
        nc_max=512,
        depth=args.depth,
        fourier_mode=args.fourier_mode,
        train_size=(args.img_size, args.img_size),
        nms_size=args.nms_size,
    ).to(device)
    M.eval()
    M.load_state_dict(
        torch.load(os.path.join("runs", name, "M.pth")),
        strict=False,
    )

    return loader, M
def random_noise_mask():
    # random smooth mask
    noise = torch.randn_like(x_plot[:, :1])
    kernel_size = 151
    xx, yy = torch.meshgrid(
        torch.arange(kernel_size),
        torch.arange(kernel_size),
        indexing="ij",
    )
    kernel = torch.exp(
        -(
            (xx - kernel_size // 2) ** 2
            + (yy - kernel_size // 2) ** 2
        )
        / (2 * (kernel_size / 6) ** 2)
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(noise.device)
    blurred_noise = F.conv2d(
        noise,
        kernel,
        padding=kernel_size // 2,
    )
    xx, yy = torch.meshgrid(
        torch.arange(noise.shape[-2]),
        torch.arange(noise.shape[-1]),
        indexing="ij",
    )
    penalization = (
        -(xx - noise.shape[-2] // 2) ** 2
        - (yy - noise.shape[-1] // 2) ** 2
    ).float()
    penalization = (
        penalization - penalization.mean()
    ) / penalization.std()
    blurred_noise = (
        blurred_noise - blurred_noise.mean()
    ) / blurred_noise.std()
    mask_plot = (
        (blurred_noise + 1 * penalization) > 0.75
    ).float()
def load_mask():
    # load mask from file
    if not os.path.exists(MASK_FILE_PATH):
        raise FileNotFoundError(f"Mask file not found at: {MASK_FILE_PATH}")
    
    # open image and convert to grayscale
    mask_img = Image.open(MASK_FILE_PATH).convert('L')
    
    # Resize mask to match x_plot dimensions (H, W)
    mask_img = resize(mask_img, [x_plot.shape[-2], x_plot.shape[-1]])
    
    # Convert to tensor
    mask_tensor = to_tensor(mask_img)
    
    # Binarize mask : threshold at 0.5 + match batch size (repeat to match dim of x_plot)
    mask_plot = (mask_tensor > 0.5).float().unsqueeze(0).repeat(x_plot.shape[0], 1, 1, 1)
    
    return mask_plot
def upscale_tensor(tensor, w, h):
    w_padded = ((ratio[1]-1)*w)
    h_padded = ((ratio[0]-1)*h)
    x_plot_upscaled = F.pad(
        tensor,
        (
            w_padded//2,
            w_padded-w_padded//2,
            h_padded//2,
            h_padded-h_padded//2,
        ),
    )
    return x_plot_upscaled

# Code
os.makedirs(save_dir, exist_ok=True)

torch.seed()

loader, M = load_model(img_idx)

# data
x_plot, mask_plot = next(iter(loader))

# Generate or load a mask
mask_plot = random_noise_mask() if USE_RANDOM_NOISE_MASK else load_mask()


# Ensure data is on correct device
x_plot = x_plot.to(device)
mask_plot = mask_plot.to(device)

# upscale
b, c, h, w = x_plot.shape
x_plot_upscaled, mask_plot_upscaled = upscale_tensor(x_plot,w,h), upscale_tensor(mask_plot,w,h)

masked_plot_upscaled = x_plot_upscaled * mask_plot_upscaled # Apply mask
n_plot_upscaled = torch.randn_like(masked_plot_upscaled) # create random noise like the masked plot

# inference
with torch.no_grad():
    out = M(
        n_plot_upscaled,
        mask_plot_upscaled,
        masked_plot_upscaled,
    )

mask_intensity = 0.6
out_vis = out.clone()
out_vis[masked_plot_upscaled.bool()] = out_vis[masked_plot_upscaled.bool()]*mask_intensity+(1.-mask_intensity)

# save
for i in range(number_of_image_batch):
    save_image(
        x_plot[i],
        os.path.join(save_dir, f"{img_idx}_original{i}.png"),
    )
    save_image(
        out_vis[i],
        os.path.join(save_dir, f"{img_idx}_outpainted{i}.png"),
    )