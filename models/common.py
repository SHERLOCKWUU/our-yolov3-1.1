# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """自动计算卷积层的相同形状的填充，选项调整膨胀。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际的卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p


class Conv(nn.Module):
    """具有批归一化和可选激活的标准 Conv2D 层，用于神经网络。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化一个标准的 Conv2D 层，包含批归一化和可选激活；参数为输入通道数、输出通道数、卷积核大小、步幅、填充、分组、膨胀和激活函数。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 激活函数

    def forward(self, x):
        """对输入 `x` 应用卷积、批归一化和激活；`x` 的形状为 [N, C_in, H, W] -> [N, C_out, H_out, W_out]。
        """
        return self.act(self.bn(self.conv(x)))  # 执行前向传播

    def forward_fuse(self, x):
        """对输入 `x` 应用融合的卷积和激活；输入形状为 [N, C_in, H, W] -> [N, C_out, H_out, W_out]。
        """
        return self.act(self.conv(x))  # 执行融合前向传播


class DWConv(Conv):
    """实现深度卷积，以便在神经网络中高效提取空间特征。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # 输入通道数, 输出通道数, 卷积核, 步幅, 膨胀, 激活
        """初始化深度卷积，具有可选的激活；参数为输入/输出通道数、卷积核、步幅、膨胀。
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)  # 调用父类构造函数


class DWConvTranspose2d(nn.ConvTranspose2d):
    """实现一个深度转置卷积层，具有指定的通道、卷积核大小、步幅和填充。"""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # 输入通道数, 输出通道数, 卷积核, 步幅, 填充, 输出填充
        """初始化深度或转置卷积层，具有指定的输入/输出通道数、卷积核大小、步幅和填充。
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))  # 调用父类构造函数


class TransformerLayer(nn.Module):
    """具有多头注意力和前馈网络的Transformer层，通过去除LayerNorm进行优化。"""

    def __init__(self, c, num_heads):
        """根据 https://arxiv.org/abs/2010.11929 初始化Transformer层，不包含LayerNorm，指定嵌入维度和头数。
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)  # 查询线性层
        self.k = nn.Linear(c, c, bias=False)  # 键线性层
        self.v = nn.Linear(c, c, bias=False)  # 值线性层
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)  # 多头注意力层
        self.fc1 = nn.Linear(c, c, bias=False)  # 前馈网络第一层
        self.fc2 = nn.Linear(c, c, bias=False)  # 前馈网络第二层

    def forward(self, x):
        """对输入张量 'x' 进行前向传播，使用多头注意力和残差连接 [batch, seq_len, features]。
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x  # 多头注意力
        x = self.fc2(self.fc1(x)) + x  # 前馈网络
        return x


class TransformerBlock(nn.Module):
    """实现一个Vision Transformer块，包含Transformer层；参考 https://arxiv.org/abs/2010.11929。"""

    def __init__(self, c1, c2, num_heads, num_layers):
        """初始化一个Transformer块，具有可选的卷积、线性和Transformer层。"""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)  # 如果输入通道数不等于输出通道数，则添加卷积层
        self.linear = nn.Linear(c2, c2)  # 可学习的位置嵌入
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))  # Transformer层序列
        self.c2 = c2  # 输出通道数

    def forward(self, x):
        """应用可选卷积，转换特征，并重塑输出以匹配输入维度。"""
        if self.conv is not None:
            x = self.conv(x)  # 如果有卷积层，则执行卷积
        b, _, w, h = x.shape  # 获取输入的形状
        p = x.flatten(2).permute(2, 0, 1)  # 展平并调整维度
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)  # 通过Transformer层处理并重塑输出

class Bottleneck(nn.Module):
    """实现一个瓶颈层，具有可选的快捷连接，用于在神经网络中有效提取特征。"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """初始化标准瓶颈层，具有可选的快捷连接；参数：输入通道 (c1)，输出通道 (c2)，快捷连接 (bool)，分组 (g)，扩展因子 (e)。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 3x3 卷积
        self.add = shortcut and c1 == c2  # 确定是否添加快捷连接

    def forward(self, x):
        """执行前向传播，进行卷积操作并可选地添加快捷连接；期望输入张量 x。
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 返回添加快捷连接或不添加的结果


class BottleneckCSP(nn.Module):
    """实现 CSP 瓶颈层，用于特征提取。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """初始化 CSP 瓶颈层，具有输入/输出通道、可选的快捷连接、分组和扩展；见
        https://github.com/WongKinYiu/CrossStagePartialNetworks。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 1x1 卷积，输入通道为 c1
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 1x1 卷积
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 1x1 卷积，将特征通道数从 2 * c_ 变为 c2
        self.bn = nn.BatchNorm2d(2 * c_)  # 对连接后的输出进行批量归一化
        self.act = nn.SiLU()  # 激活函数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))  # CSP 瓶颈层序列

    def forward(self, x):
        """通过各层处理输入，结合输出并进行激活和归一化，以实现特征提取。
        """
        y1 = self.cv3(self.m(self.cv1(x)))  # 经过 CSP 瓶颈层的输出
        y2 = self.cv2(x)  # 原始输入经过 1x1 卷积的输出
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))  # 合并输出并通过最终卷积


class CrossConv(nn.Module):
    """实现交叉卷积下采样，结合 1D 和 2D 卷积以及可选的快捷连接。"""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """初始化 CrossConv，结合 1D 和 2D 卷积的下采样选项，若输入/输出通道匹配则添加可选的快捷连接。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 1D 卷积
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 2D 卷积
        self.add = shortcut and c1 == c2  # 确定是否添加快捷连接

    def forward(self, x):
        """执行前向传播，使用顺序的 1D 和 2D 卷积以及可选的快捷连接添加。
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 返回添加快捷连接或不添加的结果

class C3(nn.Module):
    """实现一个 CSP 瓶颈层，包含 3 个卷积、可选的快捷连接、分组卷积和扩展因子。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """初始化包含 3 个卷积的 CSP 瓶颈层，具有可选的快捷连接、分组卷积和扩展因子。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv2 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv3 = Conv(2 * c_, c2, 1)  # 最后的 1x1 卷积
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))  # 瓶颈层序列

    def forward(self, x):
        """处理输入张量 `x` 通过卷积和瓶颈层，返回连接后的输出张量。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))  # 连接两个输出并通过最后的卷积


class C3x(C3):
    """扩展 C3 模块，使用交叉卷积以增强特征提取和灵活性。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化一个 C3x 模块，带有交叉卷积，扩展 C3 模块并可自定义参数。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))  # 交叉卷积序列


class C3TR(C3):
    """C3 模块，结合 TransformerBlock 在 CNN 中集成注意力机制。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化一个带有 TransformerBlock 的 C3 模块，扩展 C3 以实现注意力机制。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = TransformerBlock(c_, c_, 4, n)  # 使用 TransformerBlock


class C3SPP(C3):
    """扩展 C3 模块，结合空间金字塔池化 (SPP) 以增强 CNN 中的特征提取。"""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """初始化 C3SPP 模块，扩展 C3 模块，结合空间金字塔池化以增强特征提取。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = SPP(c_, c_, k)  # 使用空间金字塔池化


class C3Ghost(C3):
    """实现一个带有 Ghost Bottlenecks 的 C3 模块，以在神经网络中高效提取特征。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 C3Ghost 模块，使用 Ghost Bottlenecks 以高效提取特征。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))  # Ghost Bottleneck 序列


class SPP(nn.Module):
    """实现空间金字塔池化 (SPP) 以增强特征提取；见 https://arxiv.org/abs/1406.4729。"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        使用指定的通道和核初始化 SPP 层。

        更多信息见 https://arxiv.org/abs/1406.4729
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 最后的 1x1 卷积
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])  # 最大池化层

    def forward(self, x):
        """
        对输入张量 `x` 应用卷积和最大池化层，连接结果以进行特征提取。

        `x` 是形状为 [N, C, H, W] 的张量。更多细节见 https://arxiv.org/abs/1406.4729。
        """
        x = self.cv1(x)  # 经过第一个卷积
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 torch 1.9.0 max_pool2d() 的警告
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))  # 连接卷积和池化的输出

class SPPF(nn.Module):
    """实现快速空间金字塔池化 (SPPF) 层，以便在 YOLOv3 模型中高效提取特征。"""

    def __init__(self, c1, c2, k=5):  # 相当于 SPP(k=(5, 9, 13))
        """初始化 SPPF 层，指定输入/输出通道和 YOLOv3 的卷积核大小。"""
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 1x1 卷积
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化层

    def forward(self, x):
        """执行前向传播，将卷积和最大池化应用于输入 `x`，输出特征图，`x` 的形状为 [N, C, H, W]。"""
        x = self.cv1(x)  # 经过第一个卷积
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 torch 1.9.0 max_pool2d() 的警告
            y1 = self.m(x)  # 第一次最大池化
            y2 = self.m(y1)  # 第二次最大池化
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))  # 连接所有结果并通过最后的卷积


class Focus(nn.Module):
    """通过可配置卷积将空间信息集中到通道空间。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """初始化 Focus 模块，通过可配置卷积参数将宽度和高度信息集中到通道空间。"""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)  # 使用卷积将通道数乘以 4

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        """应用聚焦下采样到输入张量，返回经过卷积的输出，增加通道深度。"""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))  # 下采样并合并通道


class GhostConv(nn.Module):
    """实现 Ghost 卷积以高效提取特征；请参见 github.com/huawei-noah/ghostnet。"""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        """初始化 GhostConv，指定输入/输出通道、卷积核大小、步幅和组数；详见
        https://github.com/huawei-noah/ghostnet。
        """
        super().__init__()
        c_ = c2 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)  # 第一个卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # 第二个卷积

    def forward(self, x):
        """执行前向传播，应用卷积并连接结果；输入 `x` 是一个张量。"""
        y = self.cv1(x)  # 经过第一个卷积
        return torch.cat((y, self.cv2(y)), 1)  # 连接输出


class GhostBottleneck(nn.Module):
    """实现 Ghost Bottleneck 层，以高效提取 GhostNet 的特征。"""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        """初始化 GhostBottleneck 模块，指定输入/输出通道、卷积核大小和步幅；详见
        https://github.com/huawei-noah/ghostnet。
        """
        super().__init__()
        c_ = c2 // 2  # 隐藏通道数
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # 深度可分离卷积 (pw)
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 深度卷积 (dw)，如果步幅为 2
            GhostConv(c_, c2, 1, 1, act=False),  # 线性卷积 (pw-linear)
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )  # 快捷连接，如果步幅为 2

    def forward(self, x):
        """执行前向传播，通过网络返回卷积和快捷连接的输出之和。"""
        return self.conv(x) + self.shortcut(x)  # 连接输出和快捷连接


class Contract(nn.Module):
    """将空间维度收缩到通道，例如，将 (1,64,80,80) 收缩到 (1,256,40,40)。"""

    def __init__(self, gain=2):
        """初始化 Contract 模块，以通过指定的增益细化输入维度，例如，从 (1,64,80,80) 到 (1,256,40,40) 的默认增益为 2。"""
        super().__init__()
        self.gain = gain  # 增益因子

    def forward(self, x):
        """处理输入张量 (b,c,h,w)，收缩形状为 (b,c*s^2,h/s,w/s)，默认增益 s=2，例如，
        (1,64,80,80) 到 (1,256,40,40)。
        """
        b, c, h, w = x.size()  # 获取张量的尺寸
        s = self.gain  # 收缩因子
        x = x.view(b, c, h // s, s, w // s, s)  # 调整维度
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # 重新排列维度
        return x.view(b, c * s * s, h // s, w // s)  # 返回收缩后的张量


class Expand(nn.Module):
    """将输入张量的空间维度按因子扩展，同时相应地减少通道数。"""

    def __init__(self, gain=2):
        """初始化 Expand 模块，以按因子 `gain` 扩展空间维度，同时相应地减少通道数。"""
        super().__init__()
        self.gain = gain  # 扩展因子

    def forward(self, x):
        """将输入张量 `x` 的空间维度按因子 `gain` 扩展，同时相应地减少通道数，转换形状
        `(B,C,H,W)` 到 `(B,C/gain^2,H*gain,W*gain)`。
        """
        b, c, h, w = x.size()  # 获取张量的尺寸
        s = self.gain  # 扩展因子
        x = x.view(b, s, s, c // s**2, h, w)  # 调整维度
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # 重新排列维度
        return x.view(b, c // s**2, h * s, w * s)  # 返回扩展后的张量


class Concat(nn.Module):
    """在指定维度上连接一组张量，以高效聚合特征。"""

    def __init__(self, dimension=1):
        """初始化模块，以在指定维度上连接张量。"""
        super().__init__()
        self.d = dimension  # 连接维度

    def forward(self, x):
        """在指定维度上连接一组张量；`x` 是要连接的张量列表，默认为维度 1。"""
        return torch.cat(x, self.d)  # 连接张量


class DetectMultiBackend(nn.Module):
    """YOLOv3 multi-backend class for inference on frameworks like PyTorch, ONNX, TensorRT, and more."""

    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes multi-backend detection with options for various frameworks and devices, also handles model
        download.
        """
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a frozen TensorFlow GraphDef for inference, returning a pruned function for specified inputs
                and outputs.
                """
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Extracts and sorts non-input (output) tensor names from a TensorFlow GraphDef, excluding 'NoOp'
                prefixed tensors.
                """
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv3 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix(".pdiparams")
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv3 inference on an input image tensor, optionally with augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a Numpy array to a PyTorch tensor on the specified device, else returns the input if not a Numpy
        array.
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Warms up the model by running inference once with a dummy input of shape imgsz."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from filepath or URL, supports various formats including ONNX, PT, JIT.

        See `export_formats` for all.
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning 'stride' and 'names' if the file exists, else 'None'."""
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    """YOLOv3 模型的封装，处理多种输入类型，包括预处理、推理和非极大值抑制 (NMS)。"""

    conf = 0.25  # NMS 置信度阈值
    iou = 0.45  # NMS IoU 阈值
    agnostic = False  # NMS 类别无关
    multi_label = False  # NMS 每个框支持多个标签
    classes = None  # （可选列表）按类别过滤，例如 = [0, 15, 16] 用于 COCO 中的人、猫和狗
    max_det = 1000  # 每张图像的最大检测数量
    amp = False  # 自动混合精度 (AMP) 推理

    def __init__(self, model, verbose=True):
        """初始化推理模型，设置属性，并准备进行多线程执行，支持可选的详细日志记录。"""
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")  # 记录添加 AutoShape
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # 复制属性
        self.dmb = isinstance(model, DetectMultiBackend)  # 检查是否为 DetectMultiBackend() 实例
        self.pt = not self.dmb or model.pt  # 判断是否为 PyTorch 模型
        self.model = model.eval()  # 将模型设置为评估模式
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # 获取 Detect() 模块
            m.inplace = False  # Detect.inplace=False 确保安全的多线程推理
            m.export = True  # 不输出损失值

    def _apply(self, fn):
        """将给定函数 `fn` 应用到模型张量（不包括参数或注册缓冲区），并调整步幅和网格。"""
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # 获取 Detect() 模块
            m.stride = fn(m.stride)  # 调整步幅
            m.grid = list(map(fn, m.grid))  # 调整网格
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))  # 调整锚点网格
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """对多种输入源执行推理，支持可选的增强和性能分析；详见 `https://ultralytics.com`。"""
        #   file:        ims = 'data/images/zidane.jpg'  # 字符串或 PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR 转 RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') 或 ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (缩放到 size=640，值范围 0-1)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # 图像列表

        dt = (Profile(), Profile(), Profile())  # 记录推理时间的 Profile 实例
        with dt[0]:  # 计时
            if isinstance(size, int):  # 扩展
                size = (size, size)  # 将 size 转换为元组
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # 获取模型参数
            autocast = self.amp and (p.device.type != "cpu")  # 判断是否使用自动混合精度 (AMP) 推理
            if isinstance(ims, torch.Tensor):  # 如果输入是 torch 张量
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # 执行推理

            # 预处理
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # 图像数量，图像列表
            shape0, shape1, files = [], [], []  # 图像和推理形状，文件名
            for i, im in enumerate(ims):
                f = f"image{i}"  # 文件名
                if isinstance(im, (str, Path)):  # 文件名或 URI
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))  # 处理 EXIF 数据
                elif isinstance(im, Image.Image):  # PIL 图像
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f  # 处理 EXIF 数据
                files.append(Path(f).with_suffix(".jpg").name)  # 记录文件名
                if im.shape[0] < 5:  # 图像为 CHW 格式
                    im = im.transpose((1, 2, 0))  # 转换为 HWC 格式
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # 确保输入为 3 通道
                s = im.shape[:2]  # HWC
                shape0.append(s)  # 记录原图像形状
                g = max(size) / max(s)  # 缩放因子
                shape1.append([int(y * g) for y in s])  # 记录推理图像形状
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # 更新图像
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # 获取推理形状
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # 填充图像
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # 堆叠并从 BHWC 转为 BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 转为 fp16/32

        with amp.autocast(autocast):  # 进行推理
            # 推理过程
            with dt[1]:
                y = self.model(x, augment=augment)  # 前向传播

            # 后处理
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # 非极大值抑制
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])  # 缩放边框

            return Detections(ims, y, files, dt, self.names, x.shape)  # 返回检测结果



class Detections:
    """Handles YOLOv3 detection results with methods for visualization, saving, cropping, and format conversion."""

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """Initializes YOLOv3 detections with image data, predictions, filenames, profiling times, class names, and
        shapes.
        """
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """Executes inference on images, annotates detections, and can optionally show, save, or crop output images."""
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays image results with optional labels.

        Usage: `show(labels=True)`
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves image results with optional labels to a specified directory.

        Usage: `save(labels=True, save_dir='runs/detect/exp', exist_ok=False)`
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results; can save to `save_dir`.

        Usage: `crop(save=True, save_dir='runs/detect/exp')`.
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """
        Renders detection results, optionally displaying labels.

        Usage: `render(labels=True)`.
        """
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """Returns a copy of the detection results as pandas DataFrames for various bounding box formats."""
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """Converts Detections object to a list of individual Detection objects for iteration."""
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """Logs the string representation of the current object state to the LOGGER."""
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        """Returns the number of results stored in the instance."""
        return self.n

    def __str__(self):  # override print(results)
        """Returns a string representation of the current object state, printing the results."""
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """Returns a string representation for debugging, including class info and current object state."""
        return f"YOLOv3 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    """Implements the YOLOv3 mask Proto module for segmentation, including convolutional layers and upsampling."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        """Initializes the Proto module for YOLOv3 segmentation, setting up convolutional layers and upsampling."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs forward pass, upsampling and applying convolutions for YOLOv3 segmentation."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    """Performs image classification using YOLOv3-based architecture with convolutional, pooling, and dropout layers."""

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """Initializes YOLOv3 classification head with convolution, pooling and dropout layers for feature extraction
        and classification.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input tensor `x` through convolutions and pooling, optionally concatenating lists of tensors, and
        returns linear output.
        """
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
