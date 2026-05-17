import torch
from torch import nn
from model.FRE import common
import torch.nn.functional as F
from einops import rearrange


class TwoBranch(nn.Module):
    def __init__(self, args, num_channels = 3, num_features = 64, act = 'PReLU'):
        super(TwoBranch, self).__init__()


        modules_head_fre = [common.ConvBNReLU2D(num_channels, out_channels=num_features,
                                            kernel_size=3, stride=2, padding=1, act=act)]               
        self.head_fre = nn.Sequential(*modules_head_fre)

        modules_head_fre_mo = [common.FreBlock9_(num_features, num_features, args)
                               ]             
        self.head_fre_mo = nn.Sequential(*modules_head_fre_mo)

        modules_down1_fre = [common.DownSample(num_features, False, False),
                             common.FreBlock9(num_features, num_features, args)                               
                         ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(common.FreBlock9(num_features, num_features, args)
                                          )

        modules_down2_fre = [common.DownSample(num_features , False, False),
                         common.FreBlock9(num_features, num_features, args)
                         ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo = nn.Sequential(common.FreBlock9(num_features, num_features, args))

        modules_down3_fre = [common.DownSample(num_features, False, False),
                         common.FreBlock9(num_features, num_features, args)
                         ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(common.FreBlock9(num_features , num_features, args))

        modules_neck_fre = [common.FreBlock9_(num_features, num_features, args)
                         ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(common.FreBlock9(num_features, num_features, args))


        self.inc4_conv_1 = nn.Conv2d(num_features * 16, num_features, 1, 1, 0)
        self.outc4_conv_1 = nn.Conv2d(num_features, num_features * 16, 1, 1, 0)

        self.c4conv_fuse = common.FuseBlock7(num_features)

    
    def c4fuse_forward(self, c4, c4_FRE):
        
        c4 = self.inc4_conv_1(c4)
        c4 = self.c4conv_fuse(c4, c4_FRE)
        c4 = self.outc4_conv_1(c4)
        
        return c4

    def base_forward(self, x):

        #### fre
        x_fre = self.head_fre(x)                            #[4,64,128,128] 
        x_fre = self.head_fre_mo(x_fre)                     #[4,128,128,128]
        down1_fre = self.down1_fre(x_fre)                   #[4,256,64,64]  
        down1_fre_mo = self.down1_fre_mo(down1_fre)         #[4,256,64,64]  
        down2_fre = self.down2_fre(down1_fre_mo)            #[4,512,32,32]  
        down2_fre_mo = self.down2_fre_mo(down2_fre)         #[4,512,32,32]  
        down3_fre = self.down3_fre(down2_fre_mo)            #[4,1024,16,16]  
        down3_fre_mo = self.down3_fre_mo(down3_fre)         #[4,1024,16,16]
        neck_fre = self.neck_fre(down3_fre_mo)              #[4,2048,16,16]   
        neck_fre_mo = self.neck_fre_mo(neck_fre)                    #[4,2048,16,16]

        return  down1_fre, down2_fre, down3_fre, neck_fre_mo
    


def make_model(args, num_channels = 3, num_features = 64, act = 'PReLU'):
    return TwoBranch(args, num_channels = num_channels, num_features = num_features, act = act)

