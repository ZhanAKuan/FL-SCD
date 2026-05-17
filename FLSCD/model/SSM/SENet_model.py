
import torch.nn as nn
import torch
import model.SSM.net as net

from model.SSM.resnet import GlobalHead
from model.SSM.self_similarity import SSM

class  SENet_c1(nn.Module):
    """ResNet with Self-Similairty Encoding Module model."""

    def __init__( self, RESNET_DEPTH = 50, REDUCTION_DIM = 2048, SSM_MID_DIM = 256, UNFOLD_SIZE = 7, SSE_KERNEL_SIZE = 3):
        super(SENet_c1, self).__init__()
        print("construct SENet")
        self.RESNET_DEPTH = RESNET_DEPTH
        self.REDUCTION_DIM = REDUCTION_DIM
        
        self.SSM_MID_DIM = SSM_MID_DIM
        self.UNFOLD_SIZE = UNFOLD_SIZE
        self.SSE_KERNEL_SIZE = SSE_KERNEL_SIZE
  
        self._construct()
        self.apply(net.init_weights)

    def _construct(self):

        

        self.SSM = SSM(in_ch=self.REDUCTION_DIM, mid_ch=self.SSM_MID_DIM, unfold_size=self.UNFOLD_SIZE, ksize=self.SSE_KERNEL_SIZE)

        self.head = GlobalHead(self.REDUCTION_DIM, nc=self.REDUCTION_DIM)  
        
    def forward(self, x):

        h,w = x.shape[2], x.shape[3]
        
        A_x4 = self.SSM(x)       #[b,2048,h,w]

        x4_p = self.head.pool(A_x4)   #[b,2048,1,1]
        (b, c, _, _) = x4_p.shape

        x4_p = x4_p.view(x4_p.size(0), -1)
        x4_p = self.head.fc(x4_p)

        x_wight = x4_p.reshape([b, c, 1, 1])
        x_wight = x_wight.expand(-1, -1, h, w)
        x = torch.matmul(x_wight, x)   #[b,2048]  

        return x

