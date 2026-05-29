import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from torch import fft,einsum
from torch.nn import Module
from einops import rearrange, reduce, repeat
from functools import partial


CA=True
EMN=True



class FoMoNMS(nn.Module):
    def __init__(self, in_channels, out_channels,ref_size=(16,16) ,nms_size=3):
        super(FoMoNMS, self).__init__()
        self.nms_size=nms_size
        self.ref_size=ref_size
        self.lam=nn.Parameter(torch.ones(1,out_channels,1,1),requires_grad=True)


    def forward(self, x,m=None):
        batch, c, h, w = x.size()
        nms_size=self.nms_size
        rho_x,rho_y=h//self.ref_size[0],w//self.ref_size[1]

        ffted = torch.fft.fft2(x-x.mean(dim=(2,3), keepdim=True), dim=(-2,-1), norm='backward')              # FFT2
        phase=torch.angle(ffted)                                             # Store phase for later
        
        mod=ffted.abs()
        norm=(mod**2).sum(dim=(-2,-1),keepdim=True)**.5                      # Store for later normalization

        if EMN:
            mask = F.interpolate(m, (h, w), mode='nearest')  # (B,1,H,W)

            k = 0# optionnal erosion of the mask, to make it more robust to small misalignments. Not used in the paper, but can be useful for real-world applications.
            mask_pad = F.pad(mask, (k, k, k, k), mode='circular')
            eroded = -F.max_pool2d(-mask_pad, kernel_size=2*k+1, stride=1)
            eroded = eroded.expand_as(x)
            n1 = eroded.sum((2,3), keepdim=True).clamp(min=1)
            mean = (x * eroded).sum((2,3), keepdim=True) / n1  

            x= x*eroded -mean*eroded

        ffted = torch.fft.fft2(x, dim=(-2,-1), norm='backward')              # FFT2
        phase=torch.angle(ffted)                                             # Store phase for later
        mod=ffted.abs()
        norm=(mod**2).sum(dim=(-2,-1),keepdim=True)**.5


        mp_size=((nms_size*rho_x)//2*2+1,(nms_size*rho_y)//2*2+1)                                                # make NMS kernel size proportional to input size, and odd

        mp=nn.MaxPool2d(mp_size,stride=(1,1),padding=((nms_size*rho_x)//2,(nms_size*rho_y)//2))

        if CA:
            mean_mod=(mod*self.lam.abs()).sum(dim=(1),keepdim=True)                                                        # NMS on the mean across channels, weighted with their learnable importance.
            
            nms_mask=(mp(fft.fftshift(mean_mod,dim=(-2,-1)))==fft.fftshift(mean_mod,dim=(-2,-1)))                    # Effectively compute the NMS mask in a periodic manner.
            nms_mask=fft.ifftshift(nms_mask,dim=(-2,-1))
            nms_mask=((mp(mean_mod)==mean_mod) * nms_mask)*1                        
        else:
            nms_mask=(mp(fft.fftshift(mod,dim=(-2,-1)))==fft.fftshift(mod,dim=(-2,-1)))                    # Effectively compute the NMS mask in a periodic manner.
            nms_mask=fft.ifftshift(nms_mask,dim=(-2,-1))
            nms_mask=((mp(mod)==mod) * nms_mask)*1    


        new_mod=mod*nms_mask                                                                                     # Mask out non-maxima in the magnitude spectrum
        new_mod=new_mod*norm/(new_mod**2).sum(dim=(-2,-1),keepdim=True)**.5*m.mean(dim=(2,3),keepdim=True)**-.5  # Mask-adaptive renormalization


        new_f = new_mod*torch.exp(1.j*phase)                               # re-inject the phase
        output = fft.ifft2(new_f,dim=(-2,-1),norm='backward').real         # iFTTT2
        output = output*self.lam.abs()                                     # Importance weighting, only parameters of the module.        
        return output












class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        ''' Fourier module from Fast Fourier Convolution '''
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # FFT
        ffted=torch.view_as_real(fft.rfft2(x,dim=(-2,-1)))
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # conv -> bn -> relu
        ffted = self.conv_layer(ffted)  
        ffted = self.relu(self.bn(ffted))

        # iFFT
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  
        output= fft.irfft2(torch.view_as_complex(ffted), dim=(-2,-1))
        return output



class UFU_official(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, ffc3d=False, fft_norm='ortho',ref_size=(16,16)):
        ''' Official implementation of Unbiased Fourier Unit'''
        super(UFU_official, self).__init__()
        self.groups = groups

        self.input_shape = ref_size  # change!!!!!it!!!!!!manually!!!!!!
        self.in_channels = in_channels
        self.locMap = nn.Parameter(torch.rand(self.input_shape[0], self.input_shape[1]//2 + 1))
        self.lambda_base = nn.Parameter(torch.tensor(0.),requires_grad=True)
        self.conv_layer_down55 = torch.nn.Conv2d(in_channels=in_channels * 2 + 1, # +1 for locmap
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=False, padding_mode = 'reflect')
        self.conv_layer_down55_shift = torch.nn.Conv2d(in_channels=in_channels * 2 + 1, # +1 for locmap
                                          out_channels=out_channels * 2,
                                          kernel_size=3, stride=1, padding=2, dilation=2, groups=self.groups, bias=False, padding_mode = 'reflect')
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.img_freq = None
        self.distill = None


    def forward(self, x): 
        batch = x.shape[0]

        self.spatial_scale_factor= (x.shape[-2]//self.input_shape[0], x.shape[-1]//self.input_shape[1]) # modification
        x_copy= x*1.
        if self.spatial_scale_factor is not None: # modified from original to accomodate dynamic input sizes
            orig_size = x.shape[-2:]
            x = F.interpolate(x, size=(self.input_shape[0],self.input_shape[1]), mode=self.spatial_scale_mode, align_corners=False)


        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        locMap = self.locMap.expand_as(ffted[:,:1,:,:]) # B 1 H' W'
        ffted_copy= ffted.clone()

        cat_img_mask_freq = torch.cat((ffted[:,:self.in_channels,:,:], 
                                    ffted[:,self.in_channels:,:,:], 
                                    locMap),dim = 1)

        ffted = self.conv_layer_down55( cat_img_mask_freq )
        ffted = torch.fft.fftshift(ffted, dim = -2)

        ffted = self.relu(ffted)
        

        locMap_shift = torch.fft.fftshift(locMap, dim = -2) ## ONLY IF NOT SHIFT BACK

        # REPEAT CONV
        cat_img_mask_freq1 = torch.cat((ffted[:,:self.in_channels,:,:], 
                                    ffted[:,self.in_channels:,:,:], 
                                    locMap_shift),dim = 1)                        
        ffted = self.conv_layer_down55_shift( cat_img_mask_freq1 )
        ffted = torch.fft.ifftshift(ffted, dim = -2)



        lambda_base = torch.sigmoid(self.lambda_base)


        # irfft
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        output= x_copy*lambda_base + output*(1-lambda_base)



        epsilon = 0.5
        shift= torch.mean(output) - torch.mean(x_copy)
        output = output - shift
        output = torch.clip(output ,  float(x.min()-epsilon), float(x.max()+epsilon))

        freq_content = (output +shift -x_copy*lambda_base)/(1-lambda_base+1e-8)  # Added to vizualize the residual effect of the module
        self.distill = output

        return output,freq_content



class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale


class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)



    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """


        scale= self.scale if self.scale is not None else q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(Module):
    def __init__(
        self,
            dim,
            heads = 4,
            dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)



class ZeroOutputNN(nn.Module):
    def __init__(self):
        super(ZeroOutputNN, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x)

    

features={}
def save_activation(name):
    def hook(module, input, output):
        features[name + '_in'] = input[0].detach().clone()
        out = output[1] if isinstance(output, tuple) else output
        features[name + '_out'] = out.detach().clone()
    return hook

class BottleNeck(nn.Module):
    def __init__(self, channels, mode='None',ref_size=(16,16),nms_size=3):
        super().__init__()
        self.mode=mode
        if 'fu' in mode.lower() and not 'ufu' in mode.lower() :
            self.block=FourierUnit(channels,channels)
        elif 'ufu' in mode.lower():
            self.block=UFU_official(channels,channels,spatial_scale_factor=None,spectral_pos_encoding=True,ref_size=ref_size)
        elif 'fomo' in mode.lower():
            self.block=FoMoNMS(channels,channels,ref_size=ref_size,nms_size=nms_size)
        elif 'attention' in mode.lower():
            self.block=Attention(channels, heads=4, dim_head=32, num_mem_kv=4, flash=False)
        else:
            self.block=ZeroOutputNN()
        self.block.register_forward_hook(save_activation(f'{self.mode}'))


    def forward(self, x,m=None):
        if 'ufu' in self.mode.lower():
            out,_ = self.block(x)
            return out
        res=self.block(x,m) if 'fomo' in self.mode.lower() else self.block(x)
        return x+res
