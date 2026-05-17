import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SSM.SENet_model import SENet_c1
from model.FRE import mynet

import model.backbone.resnet_p as resnet
from model.backbone.torchutils import visulize_features

import torch
from torch import nn
import torch.nn.functional as F

class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                         replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        low_channels = 256
        high_channels = 2048

        
        self.SSM_c1 = SENet_c1(RESNET_DEPTH = 50, REDUCTION_DIM = 256, SSM_MID_DIM = 64, UNFOLD_SIZE = 7, SSE_KERNEL_SIZE = 3)
        self.two_branch = mynet.make_model(cfg, num_channels = 3, num_features = 128, act = 'PReLU')

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x1, x2, need_fp=False):
        h, w = x1.shape[-2:]

        
        FRE_feats1 = self.two_branch.base_forward(x1)
        FRE_c11, FRE_c14 = FRE_feats1[0], FRE_feats1[-1]
        FRE_feats2 = self.two_branch.base_forward(x2)
        FRE_c21, FRE_c24 = FRE_feats2[0], FRE_feats2[-1]
        
        FRE_c4 = (FRE_c14 - FRE_c24).abs()
       
        feats1 = self.backbone.base_forward(x1)
        c11, c14 = feats1[0], feats1[-1]   

        feats2 = self.backbone.base_forward(x2)  
        c21, c24 = feats2[0], feats2[-1]   #

                                 
        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs() 
      
        c4 = self.two_branch.c4fuse_forward(c4, FRE_c4)

        c1 = self.SSM_c1(c1)
    

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8        #in 2048  out 256
        rate1, rate2, rate3 = atrous_rates     #[6, 12, 18]

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
