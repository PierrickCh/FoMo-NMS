sf=1
from importlib.resources import files
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyexpat import model
from tabnanny import verbose

import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

from inception import InceptionV3
import json,re
import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import transforms
import lpips

lpips_dist = lpips.LPIPS(net='vgg').cuda()

import warnings; warnings.filterwarnings('ignore')



def center_crop_to_match(img1, img2):
    _, _, h1, w1 = img1.shape
    _, _, h2, w2 = img2.shape

    # Determine the maximum possible spatial size that both images can be cropped to
    crop_h = min(h1, h2)
    crop_w = min(w1, w2)

    # Center-crop both images to the determined size
    img1_cropped = TF.center_crop(img1, (crop_h, crop_w))
    img2_cropped = TF.center_crop(img2, (crop_h, crop_w))

    return img1_cropped, img2_cropped




    
def n_params(model):
    pp=0
    for p in list(model.parameters(True)):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

    

def get_activations(files, model, batch_size=1, dims=64,
                    cuda=False, verbose=False,sf=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([np.array(Image.open(str(f))).astype(np.float32)
                           for f in files[start:end]])

        images = images[:,:,:,0:3]
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        #batch=resize(batch,interp_method=interp.linear,scale_factors=sf)
        if cuda:
            batch = batch.cuda()
        
        pred = model(batch)[0]
       


        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)


    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


    
def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=64, cuda=False, verbose=False,sf=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose,sf=sf)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def _compute_statistics_of_path(files, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = sorted(list(path.glob('*.jpg'))+ list(path.glob('*.png')))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_sifid_given_paths(path1, path2, batch_size, cuda, dims,sf=1):
    """Calculates the SIFID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).eval()

    if cuda:
        model.cuda()

    #path1 = pathlib.Path(path1)
    files1 = path1 # sorted(list(path1.glob('*.png'))+list(path1.glob('*.jpg')))
    
    #path2 = pathlib.Path(path2)
    files2 = path2 # sorted(list(path2.glob('*.png'))+list(path2.glob('*.jpg')))
    fid_values = []
    #for i in range(len(files2)):
    m1, s1 = calculate_activation_statistics([files1], model, batch_size, dims, cuda,sf=sf)
    
    m2, s2 = calculate_activation_statistics([files2], model, batch_size, dims, cuda,sf=sf)
    fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
    return fid_values


def sifid_2imgs(im1,im2):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[64]

    model = InceptionV3([block_idx]).eval()

    if torch.cuda.is_available():
        model.cuda()

    batch_size=im1.shape[0]
    pred = model(im1)[0]
       


    act = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)
    m1 = np.mean(act, axis=0)
    s1 = np.cov(act, rowvar=False)

    pred = model(im2)[0]
    act = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)
    m2 = np.mean(act, axis=0)
    s2 = np.cov(act, rowvar=False)
    
    return calculate_frechet_distance(m1, s1, m2, s2)

def psnr(im1,im2):
    mse = torch.mean((im1 - im2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()




disp = 0

class VGG(nn.Module):
    def __init__(self, pool='max', pad=1 ):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        if False:
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
            self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=pad)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
            self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        if False:
            out['r32'] = F.relu(self.conv3_2(out['r31']))
            out['r33'] = F.relu(self.conv3_3(out['r32']))
            out['r34'] = F.relu(self.conv3_4(out['r33']))
            out['p3'] = self.pool3(out['r34'])
            out['r41'] = F.relu(self.conv4_1(out['p3']))
            out['r42'] = F.relu(self.conv4_2(out['r41']))
            out['r43'] = F.relu(self.conv4_3(out['r42']))
            out['r44'] = F.relu(self.conv4_4(out['r43']))
            out['p4'] = self.pool4(out['r44'])
            out['r51'] = F.relu(self.conv5_1(out['p4']))
            out['r52'] = F.relu(self.conv5_2(out['r51']))
            out['r53'] = F.relu(self.conv5_3(out['r52']))
            out['r54'] = F.relu(self.conv5_4(out['r53']))
        # out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

#generator's convolutional blocks 2D
class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)

    def forward(self, x):
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

#Up-sampling + batch normalization block
class Up_Bn2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8):
        super(Pyramid2D, self).__init__()

        self.cb1_1 = Conv_block2D(ch_in,ch_step)
        self.up1 = Up_Bn2D(ch_step)

        self.cb2_1 = Conv_block2D(ch_in,ch_step)
        self.cb2_2 = Conv_block2D(2*ch_step,2*ch_step)
        self.up2 = Up_Bn2D(2*ch_step)

        self.cb3_1 = Conv_block2D(ch_in,ch_step)
        self.cb3_2 = Conv_block2D(3*ch_step,3*ch_step)
        self.up3 = Up_Bn2D(3*ch_step)

        self.cb4_1 = Conv_block2D(ch_in,ch_step)
        self.cb4_2 = Conv_block2D(4*ch_step,4*ch_step)
        self.up4 = Up_Bn2D(4*ch_step)

        self.cb5_1 = Conv_block2D(ch_in,ch_step)
        self.cb5_2 = Conv_block2D(5*ch_step,5*ch_step)
        self.up5 = Up_Bn2D(5*ch_step)

        self.cb6_1 = Conv_block2D(ch_in,ch_step)
        self.cb6_2 = Conv_block2D(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv2d(6*ch_step, 3, 1, padding=0, bias=True)

    def forward(self, z):

        y = self.cb1_1(z[5])
        y = self.up1(y)
        y = torch.cat((y,self.cb2_1(z[4])),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = torch.cat((y,self.cb3_1(z[3])),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = torch.cat((y,self.cb4_1(z[2])),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = torch.cat((y,self.cb5_1(z[1])),1)
        y = self.cb5_2(y)
        y = self.up5(y)
        y = torch.cat((y,self.cb6_1(z[0])),1)
        y = self.cb6_2(y)
        y = self.last_conv(y)
        return y

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        # G.div_(h*w) # Gatys
        G.div_(h*w*c) # Ulyanov
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

# Identity function that normalizes the gradient on the call of backwards
# Used for "gradient normalization"
class Normalize_gradients(Function):
    @staticmethod
    def forward(self, input):
        return input.clone()
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.mul(1./torch.norm(grad_input, p=1))
        return grad_input,


# pre and post processing for images
if False:
    prep = transforms.Compose([
        
            #turn to BGR
            transforms.Lambda(lambda x: x[:,torch.LongTensor([2,1,0])]),
            #subtract imagenet mean
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                std=[1,1,1]),
            transforms.Lambda(lambda x: x.mul_(255)),
            ])
else: # for already loaded tensors
    prep = transforms.Compose([
        transforms.Lambda(lambda x: x[:,torch.LongTensor([2,1,0])]),
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
        ])

postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img


img_size = 512
n_input_ch = 3

vgg = VGG(pool='max', pad=1)

vgg.load_state_dict(torch.load('./vgg_conv.pth'),strict=False)
for param in vgg.parameters():
    param.requires_grad = False
vgg.cuda()
loss_layers = ['r11', 'r21', 'r31']#, 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
# these are the weights settings recommended by Gatys
# to use with Gatys' normalization:
#w = [1e2/n**3 for n in [64,128,256,512,512]]
W = [1,1,1]#,1,1]

















def build_table(lists, methods, metrics, zoom):
    # filter metrics
    metrics_filtered = [m for m in metrics ]#if m != "attention"]
    if zoom == 2:
        metrics_filtered = [m for m in metrics_filtered if (m != "psnr" and m != "lpips")]

    # header (metrics on top)
    header = ["method"] + metrics_filtered

    rows = []
    for method in methods:
        row = [method]
        for metric in metrics_filtered:
            try:
                val = torch.mean(torch.tensor(lists[(metric,method, zoom)])).item()
                row.append(f"{val:.2f}")
            except:
                row.append("-")
        rows.append(row)

    return header, rows


def print_ascii_table(header, rows, title=""):
    # compute column widths
    cols = list(zip(header, *rows))
    col_widths = [max(len(str(x)) for x in col) for col in cols]

    def format_row(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, col_widths))

    if title:
        print(title)

    print(format_row(header))
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(format_row(row))

    print()
import math

def build_table(lists, methods, metrics, zoom):
    metrics_filtered = [m for m in metrics]
    if zoom == 2:
        metrics_filtered = [m for m in metrics_filtered if (m != "psnr" and m != "lpips")]

    header = ["method"] + metrics_filtered

    rows = []
    for method in methods:
        row = [method.upper()]
        for metric in metrics_filtered:
            key = (metric, method, zoom)
            if key in lists:
                val = torch.mean(torch.tensor(lists[key])).item()
                if math.isnan(val):
                    row.append(None)  # 👈 important
                else:
                    row.append(f"{val:.2f}")
            else:
                row.append(None)
        rows.append(row)

    return header, rows


def build_ablation_table(lists, configs, metrics):
    header = ["method"] + metrics
    rows = []

    for name in configs:
        row = [name]
        for metric in metrics:
            vals = lists[(name, metric)]
            if len(vals) == 0:
                row.append(None)
            else:
                v = torch.tensor(vals).float().mean().item()
                row.append(None if math.isnan(v) else f"{v:.2f}")
        rows.append(row)

    return header, rows


def print_latex_table(header, rows, caption="", label=""):
    higher_better = {"psnr"}

    metric_names = header[1:]

    def format_metric(m):
        m_low = m.lower()

        if m_low == "gatys":
            m_clean = "GATYS ($\\times 10^3$)"
        else:
            m_clean = m.upper()

        arrow = "$\\uparrow$" if m_low in higher_better else "$\\downarrow$"
        return f"{m_clean} {arrow}"

    header_fmt = ["Method"] + [format_metric(m) for m in metric_names]

    # numeric for best
    numeric = []
    for row in rows:
        vals = []
        for j, cell in enumerate(row[1:]):
            metric = metric_names[j].lower()

            if cell is None or cell == "-":
                vals.append(None)
            else:
                val = float(cell)

                if metric == "gatys":
                    val = val / 1000.0

                vals.append(val)

        numeric.append(vals)

    best_vals = []
    for j, m in enumerate(metric_names):
        col = [numeric[i][j] for i in range(len(rows)) if numeric[i][j] is not None]

        if not col:
            best_vals.append(None)
            continue

        best_vals.append(max(col) if m.lower() in higher_better else min(col))

    n_cols = len(header_fmt)
    col_format = "l" + "c"*(n_cols-1)

    print("\\begin{table}[htb]")
    print("\\centering")
    print("\\resizebox{.8\\linewidth}{!}{%")

    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\toprule")

    print(" & ".join(header_fmt) + " \\\\")
    print("\\midrule")

    for i, row in enumerate(rows):
        method = row[0].replace("_", "")
        out_cells = [method]

        for j, cell in enumerate(row[1:]):
            metric = metric_names[j].lower()

            if cell is None or cell == "-":
                out_cells.append("")
            else:
                val = float(cell)

                if metric == "gatys":
                    display = f"{val/1000.0:.1f}"
                    val_cmp = val / 1000.0
                else:
                    display = cell
                    val_cmp = val

                if best_vals[j] is not None and abs(val_cmp - best_vals[j]) < 1e-8:
                    out_cells.append(f"\\textbf{{{display}}}")
                else:
                    out_cells.append(display)

        print(" & ".join(out_cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")

    if caption:
        print(f"\\captionof{{table}}{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")

    print("\\end{table}")
    print()


def print_latex_ablation_table(header, rows, caption="", label=""):
    higher_better = {"psnr"}

    metric_names = header[1:]

    def format_metric(m):
        m_low = m.lower()

        if m_low == "gatys":
            m_clean = "GATYS ($\\times 10^3$)"
        else:
            m_clean = m.upper()

        arrow = "$\\uparrow$" if m_low in higher_better else "$\\downarrow$"
        return f"{m_clean} {arrow}"

    header_fmt = ["Method"] + [format_metric(m) for m in metric_names]

    # ---- numeric extraction ----
    numeric = []
    for row in rows:
        vals = []

        for j, cell in enumerate(row[1:]):
            metric = metric_names[j].lower()

            if cell is None or cell == "-":
                vals.append(None)
            else:
                val = float(cell)

                if metric == "gatys":
                    val = val / 1000.0

                vals.append(val)

        numeric.append(vals)

    # ---- best per column ----
    best_vals = []
    for j, m in enumerate(metric_names):
        col = [numeric[i][j] for i in range(len(rows)) if numeric[i][j] is not None]

        if not col:
            best_vals.append(None)
            continue

        best_vals.append(max(col) if m.lower() in higher_better else min(col))

    # ---- table format ----
    n_cols = len(header_fmt)
    col_format = "l" + "c"*(n_cols-1)

    print("\\begin{table}[htb]")
    print("\\centering")
    print("\\resizebox{.8\\linewidth}{!}{%")
    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\toprule")

    print(" & ".join(header_fmt) + " \\\\")
    print("\\midrule")

    for i, row in enumerate(rows):
        method = row[0]

        method = method.replace("_", ", ")
        method = method.replace("EMN", "HPF")

        out_cells = [method]

        for j, cell in enumerate(row[1:]):
            metric = metric_names[j].lower()

            if cell is None or cell == "-":
                out_cells.append("")
            else:
                val = float(cell)

                if metric == "gatys":
                    display = f"{val/1000.0:.1f}"
                    val_cmp = val / 1000.0
                else:
                    display = cell
                    val_cmp = val

                if best_vals[j] is not None and abs(val_cmp - best_vals[j]) < 1e-8:
                    out_cells.append(f"\\textbf{{{display}}}")
                else:
                    out_cells.append(display)

        print(" & ".join(out_cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")

    if caption:
        print(f"\\captionof{{table}}{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")

    print("\\end{table}")
    print()







