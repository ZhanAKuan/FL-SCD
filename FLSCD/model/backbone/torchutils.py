import torch
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from torch.nn import MaxPool1d,AvgPool1d
from torch import Tensor
from typing import Iterable, Set, Tuple


__all__ = ['cls_accuracy']


def minmax(tensor):
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    min_, _ = tensor.min(-1, keepdim=True)
    max_, _ = tensor.max(-1, keepdim=True)
    return min_, max_

def norm_tensor(tensor,min_=None,max_=None, mode='minmax'):
    """
    输入：N*C*H*W / C*H*W / H*W
    输出：在H*W维度的归一化的与原始等大的图
    """
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    if mode == 'minmax':
        if min_ is None:
            min_, _ = tensor.min(-1, keepdim=True)
        if max_ is None:
            max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'thres':
        N = tensor.shape[-1]
        thres_a = 0.001
        top_k = round(thres_a*N)
        max_ = tensor.topk(top_k, dim=-1, largest=True)[0][..., -1]
        max_ = max_.unsqueeze(-1)
        min_ = tensor.topk(top_k, dim=-1, largest=False)[0][..., -1]
        min_ = min_.unsqueeze(-1)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)

    elif mode == 'std':
        mean, std = torch.std_mean(tensor, [-1], keepdim=True)
        tensor = (tensor - mean)/std
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'exp':
        tai = 1
        tensor = torch.nn.functional.softmax(tensor/tai, dim=-1, )
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    else:
        raise NotImplementedError
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.view(shape)

    # if tensor.ndim == 4:
    #     B, C, H, W = tensor.shape
    #     tensor = tensor.view([B, C, -1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(B, C, H, W)
    # elif tensor.ndim == 3:
    #     C, H, W = tensor.shape
    #     tensor = tensor.view([C, -1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(C, H, W)
    # elif tensor.ndim == 2:
    #     H, W = tensor.shape
    #     tensor = tensor.view([-1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(H, W)
    # else:
    #     raise NotImplementedError
def tensor2np(input_image, if_normalize=True):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy



def visulize_features(features, normalize=False):
    """
    可视化特征图，各维度make grid到一起
    """
    #from torchvision.utils import make_grid
    assert features.ndim == 4
    b,c,h,w = features.shape
    #features = features.view((b*c, 1, h, w))
    if normalize:
        features = norm_tensor(features)
    #grid = make_grid(features)
    visualize_tensors(features)

def visualize_tensors(*tensors):
    """
    可视化tensor，支持单通道特征或3通道图像
    :param tensors: tensor: C*H*W, C=1/3
    :return:
    """
    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np 
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim==2
        if tensor.ndim ==3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)

    # 创建图形
    if nums > 1:
        fig, axs = plt.subplots(1, nums, figsize=(5*nums, 5))
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
            axs[i].axis('off')  # 隐藏坐标轴
            
            # 保存单个图像
            if save_path is not None:
                plt.figure()
                plt.imshow(image, cmap='jet')
                plt.axis('off')
                plt.savefig(save_paths[i], bbox_inches='tight', pad_inches=0)
                plt.close()
                
    elif nums == 1:
        fig, ax = plt.subplots(1, nums, figsize=(5, 5))
        ax.imshow(image, cmap='jet')
        ax.axis('off')
        
        # 保存图像
        if save_path is not None:
            plt.savefig(save_paths[0], bbox_inches='tight', pad_inches=0)