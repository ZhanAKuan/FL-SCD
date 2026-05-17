import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from model.FRE.epsanet import PSAModule

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class CPCABlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        assert in_channels == out_channels

        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,21),padding=(0,10),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(21,1),padding=(10,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron       CPCA
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        tensor = tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4)
        return tensor.contiguous().view(b, -1, y // ratio, x // ratio)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out



class DownSample(nn.Module):
    def __init__(self, num_features, act, norm, out_channels = 128, scale=2):
        super(DownSample, self).__init__()
        if scale == 1:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, act=act, norm=norm, padding=1),
                ConvBNReLU2D(in_channels=num_features * scale * scale, out_channels=num_features, kernel_size=1, act=act, norm=norm)
            )
        else:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=out_channels, kernel_size=3, act=act, norm=norm, padding=1),
                invPixelShuffle(ratio=scale),
                ConvBNReLU2D(in_channels=out_channels * scale * scale, out_channels=out_channels, kernel_size=1, act=act, norm=norm)
            )

    def forward(self, inputs):
        return self.layers(inputs)

class FreBlock9(nn.Module):     #FRB模块
    def __init__(self, channels, out_channels, args):
        super(FreBlock9, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)

        
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                      nn.LeakyReLU(0.1, inplace=True),
                                      PSAModule(channels, channels))
        
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                      nn.LeakyReLU(0.1, inplace=True),
                                      PSAModule(channels, channels))
        
        
        self.post = nn.Conv2d(channels, out_channels, 1, 1, 0)

        self.cpca = CPCABlock(out_channels, out_channels)

    def forward(self, x):
       
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')       

        msF_amp = torch.abs(msF)                
        msF_pha = torch.angle(msF)              
        
        amp_fuse = self.amp_fuse(msF_amp)          
       
        amp_fuse = amp_fuse + msF_amp           

        pha_fuse = self.pha_fuse(msF_pha)       
        pha_fuse = pha_fuse + msF_pha           

        real = amp_fuse * torch.cos(pha_fuse)+1e-8  
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8  
        out = torch.complex(real, imag)+1e-8        
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = out + x
        out = self.post(out)

        out = self.cpca(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        #
        return out
    
class FreBlock9_(nn.Module):     
    def __init__(self, channels, out_channels, args):
        super(FreBlock9_, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        
        
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                      nn.LeakyReLU(0.1, inplace=True),
                                      PSAModule(channels, channels))
        
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                      nn.LeakyReLU(0.1, inplace=True),
                                      PSAModule(channels, channels))

        self.post = nn.Conv2d(channels*2, out_channels, 1, 1, 0)
        
        self.cpca = CPCABlock(out_channels, out_channels)

    def forward(self, x):
        
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')       

        msF_amp = torch.abs(msF)                
        msF_pha = torch.angle(msF)              
        
        amp_fuse = self.amp_fuse(msF_amp)          
        
        amp_fuse = amp_fuse + msF_amp           

        pha_fuse = self.pha_fuse(msF_pha)       
        pha_fuse = pha_fuse + msF_pha           

        real = amp_fuse * torch.cos(pha_fuse)+1e-8  
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8  
        out = torch.complex(real, imag)+1e-8        
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(torch.cat((out, x), dim=1))
        out = self.cpca(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
       
        return out

class Attention(nn.Module):  #SFCA
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))      #[2,512,64,64]
        k, v = kv.chunk(2, dim=1)            #[2,256,64,64] [2,256,64,64]
        q = self.q_dwconv(self.q(x))         #[2,256,64,64]

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock7(nn.Module):  
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)      
        self.spa_att = Attention(dim=channels)      
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), 
                                  nn.Conv2d(channels, 2*channels, 3, 1, 1), 
                                  nn.Sigmoid())


    def forward(self, spa, fre):
        #ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa)+fre   
        spa = self.spa_att(spa, fre)+spa   
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res
