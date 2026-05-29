from PIL import Image
import torch
from torchvision.utils import make_grid,save_image
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
import random, itertools
import argparse, os, json
from pathlib import Path
import fourier_modules as FM
from tqdm import tqdm

device='cuda:0'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


    
def n_params(model):
    pp=0
    for p in list(model.parameters(True)):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def load_args(config_file, old_args):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            new_args = json.load(f)
        for key, value in new_args.items():
            setattr(old_args, key, value)
        return old_args
    else:
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    


conv2d=nn.Conv2d
#def conv2d(*args, **kwargs):
#    return spectral_norm(nn.Conv2d(*args, **kwargs))
Norm=nn.BatchNorm2d
DNorm=nn.InstanceNorm2d



def GP(netD, real_data, fake_data, device='cuda:0'):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    outs = netD(interpolates)
    disc_interpolates = torch.cat([out.view(out.shape[0], -1) for out in outs], dim=-1)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self,nc):
        super().__init__()

        self.weight = nn.Parameter(0.1*torch.randn(1,nc,1,1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            noise = torch.randn(feat.shape).to(feat.device)
        return feat + self.weight * noise


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False,padding_mode='circular'),
        NoiseInjection(out_planes*2),
        Norm(out_planes*2), 
        GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False,padding_mode='circular'),
        NoiseInjection(out_planes*2),
        Norm(out_planes*2), GLU()
        )
    return block



class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False,padding_mode='circular'),
            Norm(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False,padding_mode='circular'),
            Norm(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False,padding_mode='circular'),
            Norm(out_planes),
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

class Disc_DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Disc_DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False,padding_mode='circular'),
            DNorm(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False,padding_mode='circular'),
            DNorm(out_planes),
            nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False,padding_mode='circular'),
            DNorm(out_planes), 
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False,padding_mode='circular'), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False,padding_mode='circular'), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)
    




class Enc(nn.Module):
    def __init__(self,nc_im=3,nc_start=8,nc_max=128,depth=5):
        super(Enc,self).__init__()
        l=[nn.Sequential(nn.Conv2d(nc_im,int(nc_start),3,1,1,padding_mode='circular'),nn.LeakyReLU(.2))]
        l+=[DownBlockComp(min(nc_start*2**i,nc_max),min(nc_start*2**(i+1),nc_max)) for i in range(depth)]
        self.main=nn.ModuleList(l)
    def forward(self,x):
        x=x*2-1
        for m in self.main:
            x=m(x)
        return x
        


        
class Dec(nn.Module):
    def __init__(self,nc_im=3,nc_start=8,nc_max=128,depth=5):
        super(Dec,self).__init__()
        l=[nn.Conv2d(int(nc_start),nc_im,3,1,1,padding_mode='circular')]
        l+=[UpBlockComp(min(nc_start*2**(i+1),nc_max),min(nc_start*2**i,nc_max)) for i in range(depth)]
        self.main=nn.ModuleList(l[::-1])
    def forward(self,x):
        for m in self.main:
            x=m(x)
        return x



class Model(nn.Module):
    def __init__(self,nc_im=3,nc_start=16,nc_max=128,depth=4,fourier_mode='NMSFU_no_freq',train_size=(256,256),nms_size=8):
        super(Model,self).__init__()
        self.enc=Enc(nc_im=nc_im*2+1,nc_start=nc_start,nc_max=nc_max,depth=depth)
        self.dec=Dec(nc_im=nc_im,nc_start=nc_start,nc_max=nc_max,depth=depth)
        ref_size=(train_size[0]//2**depth,train_size[1]//2**depth)
        self.Fourier_module=FM.BottleNeck(channels=nc_start*2**depth,mode=fourier_mode,ref_size=ref_size,nms_size=nms_size)
        

    def forward(self,n,mask,masked):
        
        x=self.enc(torch.cat((n,masked,mask),dim=1))
        x=self.Fourier_module(x,mask)
        x=self.dec(x)
        return torch.sigmoid(x)
    



class Disc(nn.Module):
    def __init__(self,nc_im=3,nc_start=16,nc_max=128,depth=2):
        super(Disc,self).__init__()
        l=[nn.Sequential(conv2d(nc_im,int(nc_start),3,1,1,padding_mode='circular'),nn.LeakyReLU(.2))]
        l+=[Disc_DownBlockComp(min(nc_start*2**i,nc_max),min(nc_start*2**(i+1),nc_max)) for i in range(depth)]
        self.main=nn.ModuleList(l)
        self.tail=conv2d(min(nc_start*2**(depth),nc_max),1,1,padding_mode='circular')

    def forward(self,x):
        for i,m in enumerate(self.main):
            x=m(x)
        return self.tail(x)
    





class DS_rot(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg','jpeg','png','tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
        octaves=1,
        scales=1,
        min_mask_shape=(16,16),
        max_mask_shape=(64,64),
        delta_angle=5
    ):
        '''made for single image dataset, augment with rotations, precomputing the rotated versions of the image, and taking random crops.'''
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.min_mask_shape = min_mask_shape
        self.max_mask_shape = max_mask_shape
        self.to_tensor = T.ToTensor()
        self.paths = [p for ext in exts for p in Path(folder).glob(f"**/*.{ext}")]
        self.images = []
        angles = list(range(-180, 181,delta_angle))
        rot_scales = [1]   # you had scale = 1
        for path in self.paths:
            img = Image.open(path).convert("RGB")
            Rw, Rh = img.size
            crop_dim = int(0.6464 * rot_scales[0] * min(Rw, Rh)) #avoid black borders after rotation, 0.6464 is the factor for 45 degree rotation, which is the worst case scenario
            for angle, sc in itertools.product(angles, rot_scales):
                aug = TF.affine(img,angle=angle,translate=(0, 0),scale=sc,shear=0,interpolation=TF.InterpolationMode.BILINEAR)
                cropped = TF.center_crop(aug, crop_dim)
                self.images.append(self.to_tensor(cropped))
        self.crop = T.RandomCrop((image_size[-2], image_size[-1]))
        self.scale_factors = [torch.tensor(1.0).half()]

    def __len__(self):
        return 1000000  

    def __getitem__(self, index):
        index = index % len(self.images)
        img = self.images[index]
        img = self.crop(img)
        _, H, W = img.shape
        if H < self.image_size[-2] or W < self.image_size[-1]:
            raise ValueError("Crop size larger than image size after preprocessing.")
        mask = torch.zeros((1, self.image_size[-2], self.image_size[-1]))

        mh = torch.randint(
            self.min_mask_shape[0],
            min(self.max_mask_shape[0], self.image_size[-2]) + 1,
            (1,)
        ).item()

        mw = torch.randint(
            self.min_mask_shape[1],
            min(self.max_mask_shape[1], self.image_size[-1]) + 1,
            (1,)
        ).item()
        margin=3
        top = torch.randint(
            margin,
            self.image_size[-2] - mh + 1-margin,
            (1,)
        ).item()

        left = torch.randint(
            margin,
            self.image_size[-1] - mw + 1 -margin,
            (1,)
        ).item()

        mask[:, top:top+mh, left:left+mw] = 1.0
        return img, mask 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the saved args.json file')
    parser.add_argument('--dataset_path', default='./images/data/fabric_512/', type=str, help='dataset path')
    parser.add_argument('--name', type=str, default='test', help='name of the results folder')
    
    parser.add_argument('--lr', default= 1e-4, type=float, help='learning rate')
    parser.add_argument('--bs', default= 16, type=int, help='batch size')
    parser.add_argument('--fourier_mode', default= 'NMSFU_no_freq', type=str, help='choice of residual connexion in bottlenck')
    parser.add_argument('--CA', default= True, type=str2bool, help='use channel Alignment')
    parser.add_argument('--EMN', default= True, type=str2bool, help='use high pass filter')
    parser.add_argument('--nms_size', default= 5, type=int, help='NMS hyperparameter')
    parser.add_argument('--dim', default=16, type=int, help='base dimension of UNET')
    parser.add_argument('--img_size', default=256,  type=int, help='size of crops during training')
    parser.add_argument('--lam_gan',default=1e-2,type=float)
    parser.add_argument('--depth',default=2,type=int)
    
    parser.add_argument('--training_steps', default=100000, type=int, help='steps every job')
    parser.add_argument('--save_every', default=10000, type=int, help='save_every')
    parser.add_argument('--seed', default=0., type=float, help='seed')

    args = parser.parse_args()
    os.makedirs('./runs/%s'%args.name,exist_ok=True)
    with open('./runs/%s/args.json'%args.name, 'w+') as f:
        json.dump(vars(args), f, indent=4)
    torch.manual_seed(args.seed)
    print(args.name)
    FM.CA=args.CA
    FM.EMN=args.EMN
    dset=DS_rot(args.dataset_path,image_size=(args.img_size,args.img_size),min_mask_shape=(64,64),max_mask_shape=(200,200))
    loader=DataLoader(dset,batch_size=args.bs,shuffle=True,drop_last=True,num_workers=0)
    M = Model(nc_im=3,nc_start=args.dim,nc_max=512,depth=args.depth,fourier_mode=args.fourier_mode,train_size=(args.img_size,args.img_size),nms_size=args.nms_size).to(device)
    D = Disc(nc_im=3,nc_start=args.dim,nc_max=512,depth=args.depth).to(device)
    print('M',n_params(M)/10**6,'D',n_params(D)/10**6)

    dir='./runs/%s'%args.name
    dir_inf=os.path.join(dir,'inference')
    dir_inf2x=os.path.join(dir,'inference2x')
    os.makedirs(dir,exist_ok=True)
    os.makedirs(dir_inf,exist_ok=True)
    os.makedirs(dir_inf2x,exist_ok=True)

    try:
        #M.load_state_dict(torch.load(os.path.join(dir,'M.pth')),strict=False)
        print('model NOT loaded!')
    except:
        pass



    opt=optim.Adam(M.parameters(),lr=args.lr) 
    opt_D=optim.Adam(D.parameters(),lr=args.lr)
    x_plot,mask_plot=next(iter(loader))


    x_plot,mask_plot=x_plot.to(device),mask_plot.to(device)
    masked_plot=x_plot*mask_plot
    n_plot=torch.randn_like(masked_plot)
    b,c,h,w=x_plot.shape
    x_plot2x=F.pad(x_plot,(w//2,w-w//2,h//2,h-h//2))
    mask_plot2x=F.pad(mask_plot,(w//2,w-w//2,h//2,h-h//2))
    masked_plot2x=x_plot2x*mask_plot2x
    n_plot2x=torch.randn_like(masked_plot2x)



    for i in tqdm(range(args.training_steps)):
        x,mask=next(iter(loader))
        x,mask=x.to(device),mask.to(device)
        masked=x*mask
        n=torch.randn_like(masked).to(device)


        rec=M(n,mask,masked)
        M.zero_grad()
        D.eval()
        loss_D=D(rec).mean()
        loss_rec=  ((mask*.9+.1)*(x-rec)**2).mean()  # L2 loss mostly inside mask
        (0.1*loss_rec+args.lam_gan*loss_D).backward()

        opt.step()

        D.train()
        D.zero_grad()
        D_f=D(rec.detach())
        D_r=D(x)

        loss_D=(D_r-D_f).mean()
        (loss_D+.1*GP(D,rec.detach(),x,device=device)).backward(retain_graph=True)
        
        opt_D.step()
        

    
        
        if ((i+1)%args.save_every)==0: # save intermediate results
            with torch.no_grad():
                rec=M(n_plot,mask_plot,masked_plot)
                im= torch.cat((make_grid(x_plot+.3*mask_plot),make_grid(rec)),dim=-2).cpu()
                save_image(im,os.path.join(dir,'inference','train_%d.png'%i))#,'%d.png'% epoch))

                if 'fomo' in args.fourier_mode.lower(): 
                    lam=M.Fourier_module.block.lam 
                    _,idx=torch.sort(lam.abs(),dim=1,descending=True)



                f_in=FM.features['%s_in'%args.fourier_mode]
                f=FM.features['%s_out'%args.fourier_mode]
                if 'fomo' in args.fourier_mode.lower():
                    f=f_in[:,idx.view(-1)]
                fmin,fmax=f.view(f.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),f.view(f.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                f_in=(f-fmin)/(fmax-fmin)
                m=torch.log1p(fft.fftshift(fft.fft2(f-f.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1),norm='backward'),dim=(-2,-1)).abs())
                m=(m-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/(m.view(m.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                m_in=1.*m

                f=FM.features['%s_out'%args.fourier_mode]
                if 'fomo' in args.fourier_mode.lower():
                    f=f[:,idx.view(-1)]
                fmin,fmax=f.view(f.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),f.view(f.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                norm=torch.max(-fmin,fmax)
                f_out=(f-fmin)/(fmax-fmin)

                m=torch.log1p(fft.fftshift(fft.fft2(f-f.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1),norm='backward'),dim=(-2,-1)).abs())
                m_out=(m-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/(m.view(m.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                fm=torch.cat((torch.cat([f_in,f_out],dim=-2),torch.cat([m_in,m_out],dim=-2)),dim=-1)
                fm=F.interpolate(fm,size=(rec.shape[-2]//1,2*x_plot.shape[-1]//2),mode='nearest')
                B=8
                for b in range(B):
                    
                    i1 = make_grid(
                        rec[b:b+1] + mask_plot[b:b+1] * 0.2,
                        nrow=1,
                        padding=0
                    )

                    i2 = make_grid(
                        torch.cat([fm[b, i] for i in range(6)], dim=-1)
                            .unsqueeze(0)    
                            .unsqueeze(1)    
                            .repeat(1, 3, 1, 1),
                        nrow=1,
                        padding=0
                    )


                
                    grid = torch.cat([i1, i2], dim=-1).permute(0,2,1)
                    os.makedirs(os.path.join(dir_inf,str(b)),exist_ok=True)
                    save_image(
                        grid.permute(0, 2, 1),
                        os.path.join(dir_inf,str(b), f"{i}.png")
                    )


                rec=M(n_plot2x,mask_plot2x,masked_plot2x)
                im= torch.cat((make_grid(x_plot2x+.3*mask_plot2x),make_grid(rec)),dim=-2).cpu()
                save_image(im,os.path.join(dir,'inference2x','train_%d.png'%i))#,'%d.png'% epoch))






                f_in=FM.features['%s_in'%args.fourier_mode]
                f=FM.features['%s_out'%args.fourier_mode]
                if 'fomo' in args.fourier_mode.lower():
                    f=f_in[:,idx.view(-1)]
                fmin,fmax=f.view(f.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),f.view(f.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                f_in=(f-fmin)/(fmax-fmin)
                m=torch.log1p(fft.fftshift(fft.fft2(f-f.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1),norm='backward'),dim=(-2,-1)).abs())
                m=(m-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/(m.view(m.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                m_in=1.*m

                f=FM.features['%s_out'%args.fourier_mode]
                if 'fomo' in args.fourier_mode.lower():
                    f=f[:,idx.view(-1)]
                fmin,fmax=f.view(f.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),f.view(f.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                norm=torch.max(-fmin,fmax)
                f_out=(f-fmin)/(fmax-fmin)

                m=torch.log1p(fft.fftshift(fft.fft2(f-f.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1),norm='backward'),dim=(-2,-1)).abs())
                m_out=(m-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/(m.view(m.shape[0],-1).max(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-m.view(m.shape[0],-1).min(dim=(-1))[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                fm=torch.cat((torch.cat([f_in,f_out],dim=-2),torch.cat([m_in,m_out],dim=-2)),dim=-1)
                fm=F.interpolate(fm,size=(rec.shape[-2]//1,2*x_plot2x.shape[-1]//2),mode='nearest')
                B=8
                for b in range(B):
                    
                    i1 = make_grid(
                        rec[b:b+1] + mask_plot2x[b:b+1] * 0.2,
                        nrow=1,
                        padding=0
                    )


                    i2 = make_grid(
                        torch.cat([fm[b, i] for i in range(6)], dim=-1)
                            .unsqueeze(0)  
                            .unsqueeze(1)  
                            .repeat(1, 3, 1, 1),
                        nrow=1,
                        padding=0
                    )


                
                    grid = torch.cat([i1, i2], dim=-1).permute(0,2,1)
                    os.makedirs(os.path.join(dir_inf2x,str(b)),exist_ok=True)
                    save_image(
                        grid.permute(0, 2, 1),
                        os.path.join(dir_inf2x,str(b), f"{i}.png")
                    )


            torch.save(M.state_dict(), os.path.join(dir,'M.pth'))
    torch.save(M.state_dict(), os.path.join(dir,'M.pth'))
