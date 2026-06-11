# code from Mahé DUVAL
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

import gradio as gr
import numpy as np

from demos.theme import *
# Parameters
USE_RANDOM_NOISE_MASK = False  # Set to False to load from a file instead
MASK_FILE_PATH = "demos/mask_05.jpg"  # Path to the mask file

dir_fig = 'demos/images/figures/'
save_dir = "demos/images/figures/complex_masks_inpainting"

method = "fomonms_nms_size_3"

img_idx = 8 # reference image to use
number_of_image_batch = 1 # number of images to create (batches)

# Functions
def load_model(image_idx, method_name)-> DataLoader | Model:
    # find run
    
    file_arg = ""
    
    match method_name:
        case "FoMo-NMS":
            file_arg = "fomonms_size_3_EMN_1_CA_1"
        case "UFU":
            file_arg = "UFU_official"
        case "FU":
            file_arg = "FU"
        case "Vanilla":
            file_arg = "Vanilla"
        case _ :
            file_arg = ""
    
    for d in os.listdir("./runs"):
        if str(image_idx) == d.split("_")[1] and file_arg in d:
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
def random_noise_mask(reference_tensor):
    # random smooth mask
    noise = torch.randn_like(reference_tensor[:, :1])
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
def load_image_mask(ref_tensor):
    # load mask from file
    if not os.path.exists(MASK_FILE_PATH):
        raise FileNotFoundError(f"Mask file not found at: {MASK_FILE_PATH}")
    
    # open image and convert to grayscale
    mask_img = Image.open(MASK_FILE_PATH).convert('L')
    
    # Resize mask to match the reference tensor dimensions (H, W)
    mask_img = resize(mask_img, [ref_tensor.shape[-2], ref_tensor.shape[-1]])
    
    # Convert to tensor
    mask_tensor = to_tensor(mask_img)
    
    # Binarize mask : threshold at 0.5 + match batch size (repeat to match dim of the reference tensor)
    mask_plot = (mask_tensor > 0.5).float().unsqueeze(0).repeat(ref_tensor.shape[0], 1, 1, 1)
    
    return mask_plot
def load_np_mask(np_img, ref_tensor):
    # numpy image to PIL and convert to grayscale
    mask_img = Image.fromarray(np_img).convert('L')
    
    # Resize mask to match the reference tensor dimensions (H, W)
    mask_img = resize(mask_img, [ref_tensor.shape[-2], ref_tensor.shape[-1]])
    
    # Convert to tensor
    mask_tensor = to_tensor(mask_img)
    
    # Binarize mask : threshold at 0.5 + match batch size (repeat to match dim of the reference tensor)
    mask_plot = (mask_tensor > 0.5).float().unsqueeze(0).repeat(ref_tensor.shape[0], 1, 1, 1)
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
def process_fomo_original(img_idx, save=False):
    torch.seed()

    loader, M = load_model(img_idx, "FoMo-NMS")

    # data
    x_plot, mask_plot = next(iter(loader))

    # Generate or load a mask
    mask_plot = random_noise_mask(x_plot) if USE_RANDOM_NOISE_MASK else load_image_mask(x_plot)


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
    if not save : return x_plot, out_vis
    
    for i in range(number_of_image_batch):
        save_image(
            x_plot[i],
            os.path.join(save_dir, f"{img_idx}_original{i}.png"),
        )
        save_image(
            out_vis[i],
            os.path.join(save_dir, f"{img_idx}_outpainted{i}.png"),
        )

    return x_plot, out_vis
def process_outpainting(img_idx, img_mask,display_function,method_name,save=False):
    torch.seed()

    loader, M = load_model(img_idx,method_name)

    # data
    x_plot, mask_plot = next(iter(loader))

    # Generate or load a mask
    mask_plot = load_np_mask(img_mask, x_plot)


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

    out_vis = display_function(out, mask_plot_upscaled)
    
    
    # save
    if not save : return x_plot, out_vis
    
    for i in range(number_of_image_batch):
        save_image(
            x_plot[i],
            os.path.join(save_dir, f"{img_idx}_original{i}.png"),
        )
        save_image(
            out_vis[i],
            os.path.join(save_dir, f"{img_idx}_outpainted{i}.png"),
        )

    return x_plot, out_vis

## Display functions
def overlay(out, mask):
    mask_intensity = 0.6
    out_vis = out.clone()
    mask = mask.expand_as(out) # to RGB
    out_vis[mask.bool()] = out_vis[mask.bool()]*mask_intensity+(1.-mask_intensity)

    return out_vis

def border(out, mask):
    k = 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=2*k+1,
        stride=1,
        padding=k,
    )

    eroded = -F.max_pool2d(
        -mask,
        kernel_size=2*k+1,
        stride=1,
        padding=k,
    )
    
    border = ((dilated - eroded) > 0.5).float()

    # expand border to RGB
    border = border.expand_as(out)

    # keep original range, only overwrite border pixels with black
    out_vis = out.clone()
    out_vis[border.bool()] = 1.-out_vis[border.bool()] # use inverse instead for better visibility on dark pictures
    
    return out_vis

## Utilities
def tensor_to_img(tensor:torch.Tensor): # adapted from save_image() in ./torchvision/utils.py 
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im

# Specific interface functions
def process_fomo_gradio_interface(ref_img_idx,input_img):
    inversed_mask = np.abs(255 - input_img['composite'])
    original_img, outpainted_img = process_outpainting(ref_img_idx,inversed_mask, visualization_function,"FoMo-NMS")
    return tensor_to_img(outpainted_img[0])
def process_ufu_gradio_interface(ref_img_idx,input_img):
    inversed_mask = np.abs(255 - input_img['composite'])
    original_img, outpainted_img = process_outpainting(ref_img_idx,inversed_mask, visualization_function,"UFU")
    return tensor_to_img(outpainted_img[0])
def process_fu_gradio_interface(ref_img_idx,input_img):
    inversed_mask = np.abs(255 - input_img['composite'])
    original_img, outpainted_img = process_outpainting(ref_img_idx,inversed_mask, visualization_function,"FU")
    return tensor_to_img(outpainted_img[0])
def process_vanilla_gradio_interface(ref_img_idx,input_img):
    inversed_mask = np.abs(255 - input_img['composite'])
    original_img, outpainted_img = process_outpainting(ref_img_idx,inversed_mask, visualization_function,"Vanilla")
    return tensor_to_img(outpainted_img[0])


def update_ref_img_preview(idx):
    return f"images/data/{idx}/{idx}.png"
## Globals
visualization_function = overlay
def update_visualization_mode(i):
    global visualization_function
    match i:
        case "Border" : 
            visualization_function = border
        case "Overlay" : 
            visualization_function = overlay

ratio = (2, 2) # upscale ratio
def update_scale_ratio(i):
    global ratio
    ratio = (i, i)

# Interface
with gr.Blocks() as demo : 
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    with gr.Row(equal_height=True):
        im_layers = gr.LayerOptions(layers=["Mask"], allow_additional_layers=False)
        im_brushes = gr.Brush(default_size="auto", colors=["rgb(0, 0, 0)"])
        
        in_canvas = gr.ImageEditor(
            type="numpy",
            label="Input",
            sources=(),
            elem_classes="full_height",
            elem_id="image_canvas",
            canvas_size=(512,512),
            brush=im_brushes,
            height=350
            
        )

        out_im_preview = gr.Image(
            type="numpy",
            label="Output",
            elem_classes="output-image-fill",
            interactive=False,
            height=350
        )
    with gr.Row(equal_height=True):
        in_vis_mode = gr.Dropdown(
            choices=["Border","Overlay"],
            label="Zone visualization mode",
            value="Overlay"
        )
        in_upscale_ratio = gr.Slider(
            minimum= 1,
            maximum= 10,
            value = 2,
            step = 1,
            label = "Upscale ratio"
        )
    force_generate_btn = gr.Button("Force Generate", variant="primary")
    with gr.Accordion("Other Methods"):
        with gr.Row(equal_height=True):
            out_preview_img_FU = gr.Image(
                type="numpy",
                label="Output - FU",
                elem_classes="output-image-fill fixed_height_300",
                interactive=False,
            )
            out_preview_img_UFU = gr.Image(
                type="numpy",
                label="Output - UFU",
                elem_classes="output-image-fill fixed_height_300",
                interactive=False,
            )
            out_preview_img_Vanilla = gr.Image(
                type="numpy",
                label="Output - Vanilla",
                elem_classes="output-image-fill fixed_height_300",
                interactive=False,
            )
            
    with gr.Accordion("Reference Image", True, ):
        with gr.Column():
            out_preview_ref_img = gr.Image(
                value=f"images/data/{img_idx}/{img_idx}.png",
                type="filepath",
                elem_classes="output-image-fill",
                height=250
            )
            in_ref_img_idx = gr.Slider(
                minimum = 1,
                maximum = 15,
                value = img_idx,
                step = 1,
                label = "Reference Image", 
                
            )
    update_canvas = in_canvas.change(
        fn = process_fomo_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_im_preview],
        show_progress=True
    ).then(
        fn = process_fu_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_FU],
        show_progress=True
    ).then(
        fn = process_ufu_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_UFU],
        show_progress=True
    ).then(
        fn = process_vanilla_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_Vanilla],
        show_progress=True
    )
    
    
    in_vis_mode.change(
        fn = update_visualization_mode,
        inputs=in_vis_mode
    )
    in_upscale_ratio.change(
        fn = update_scale_ratio,
        inputs = in_upscale_ratio
    )
    
    in_ref_img_idx.change(
        fn = update_ref_img_preview,
        inputs = in_ref_img_idx,
        outputs = out_preview_ref_img,
        show_progress=False
    )
    
    force_generate_btn.click(
        fn = process_fomo_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_im_preview],
        show_progress=True
    ).then(
        fn = process_fu_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_FU],
        show_progress=True
    ).then(
        fn = process_ufu_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_UFU],
        show_progress=True
    ).then(
        fn = process_vanilla_gradio_interface,
        inputs=[in_ref_img_idx,in_canvas],
        outputs=[out_preview_img_Vanilla],
        show_progress=True
    )
    gr.HTML(HTML_FOOTER)
# Launch the demo

demo.queue().launch(theme=theme, css=CUSTOM_CSS, head=HTML_CUSTOM_HEAD)