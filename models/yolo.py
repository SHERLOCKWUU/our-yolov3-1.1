# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

# 获取当前文件的绝对路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到系统路径
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径（非Windows系统）

# 导入自定义模块和YOLOv3依赖
from models.common import *  # noqa
from models.experimental import *  # noqa
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

# 尝试导入FLOPs（浮点运算量）的计算模块thop
try:
    import thop  # 用于计算FLOPs
except ImportError:
    thop = None


class Detect(nn.Module):
    """YOLOv3的检测头类，负责处理模型输出，包括网格和锚点网格的生成。"""

    stride = None  # 步长，在构建过程中计算
    dynamic = False  # 强制重新生成网格
    export = False  # 导出模式标志

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # 检测层
        """初始化YOLOv3检测层，包括类别数量、锚点、通道数和是否原地操作（inplace）选项。"""
        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚点的输出数量
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # 每个检测层的锚点数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化锚点网格
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # 注册锚点张量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        self.inplace = inplace  # 是否原地操作（例如切片赋值）

    def forward(self, x):
        """
        处理输入，通过卷积层并重新调整输出用于检测。
        输入 x 是形状为(bs, C, H, W)的张量列表。
        """
        z = []  # 推理输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积操作
            bs, _, ny, nx = x[i].shape  # 形状从 x(bs,255,20,20) 变为 x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理阶段
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # 检测并生成分割掩码
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # 计算xy坐标
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # 计算宽高wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # 仅检测框
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # 计算xy坐标
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # 计算宽高wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """生成网格和锚点网格，形状为 `(1, num_anchors, ny, nx, 2)`，用于锚点索引。"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # 网格形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # 兼容torch>=0.7
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # 添加网格偏移，即y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv3的分割头，用于分割模型，增加了掩码预测和原型功能。"""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """初始化分割头，包含可配置的类别数量、锚点、掩码、原型、通道和是否原地操作选项。"""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.no = 5 + nc + self.nm  # 每个锚点的输出数量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型层
        self.detect = Detect.forward

    def forward(self, x):
        """执行前向传播，返回预测结果和原型，根据训练或导出状态返回不同的输出。"""
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """实现用于目标检测任务的基础 YOLOv3 模型架构。"""

    def forward(self, x, profile=False, visualize=False):
        """对输入 `x` 执行单尺度推理或训练步骤，可选择进行性能分析和可视化。"""
        return self._forward_once(x, profile, visualize)  # 单尺度推理，训练

    def _forward_once(self, x, profile=False, visualize=False):
        """执行单次推理或训练步骤，提供性能分析和可视化选项以处理输入 `x`。"""
        y, dt = [], []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自之前的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """通过测量执行时间和计算成本来分析模型的单层性能。"""
        c = m == self.model[-1]  # 是否为最后一层，复制输入以修正inplace操作
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # 融合 Conv2d() + BatchNorm2d() 层
        """融合模型中的 Conv2d() 和 BatchNorm2d() 层以优化推理速度。"""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新 conv
                delattr(m, "bn")  # 移除 batchnorm
                m.forward = m.forward_fuse  # 更新 forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # 打印模型信息
        """打印模型信息；`verbose` 用于详细输出，`img_size` 用于输入图像大小（默认 640）。"""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """将 `to()`、`cpu()`、`cuda()`、`half()` 应用于模型张量，排除参数或已注册缓冲区。"""
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv3 检测模型类，用于初始化和处理带有可配置参数的检测模型。"""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):  # 模型，输入通道数，类别数
        """初始化 YOLOv3 检测模型，带有可配置的 YAML、输入通道、类别和 anchors。"""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 模型字典
        else:  # 是 *.yaml 文件
            import yaml  # 用于 torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # 加载模型字典

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"覆盖 model.yaml 中的 nc={self.yaml['nc']} 为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖 yaml 中的类别数
        if anchors:
            LOGGER.info(f"覆盖 model.yaml 中的 anchors 为 anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # 覆盖 yaml 中的 anchors
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 模型，保存列表
        self.names = [str(i) for i in range(self.yaml["nc"])]  # 默认类别名称
        self.inplace = self.yaml.get("inplace", True)

        # 构建 strides 和 anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x 最小 stride
            m.inplace = self.inplace

            def forward(x):
                """通过模型传递输入 'x' 并返回处理后的输出。"""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # 前向传播
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # 仅运行一次

        # 初始化权重和偏差
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """通过模型处理输入，提供增强、性能分析和可视化的选项。"""
        if augment:
            return self._forward_augment(x)  # 增强推理，None
        return self._forward_once(x, profile, visualize)  # 单尺度推理，训练

    def _forward_augment(self, x):
        """通过缩放和翻转输入图像来执行增强推理，返回拼接后的预测结果。"""
        img_size = x.shape[-2:]  # 高度，宽度
        s = [1, 0.83, 0.67]  # 缩放比例
        f = [None, 3, None]  # 翻转 (2-上下，3-左右)
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # 前向传播
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # 保存
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 剪裁增强推理的尾部
        return torch.cat(y, 1), None  # 增强推理，训练

    def _descale_pred(self, p, flips, scale, img_size):
        """在增强后调整预测的缩放和翻转，基于图像尺寸调整比例和翻转。"""
        if self.inplace:
            p[..., :4] /= scale  # 取消缩放
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # 取消上下翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # 取消左右翻转
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # 取消缩放
            if flips == 2:
                y = img_size[0] - y  # 取消上下翻转
            elif flips == 3:
                x = img_size[1] - x  # 取消左右翻转
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """从 YOLOv3 预测结果中剪除增强推理的尾部，主要影响第一个和最后一个检测层。"""
        nl = self.model[-1].nl  # 检测层的数量（P3-P5）
        g = sum(4 ** x for x in range(nl))  # 网格点数
        e = 1  # 排除的层数
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # 索引
        y[0] = y[0][:, :-i]  # 大物体
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][:, i:]  # 小物体
        return y

    def _initialize_biases(self, cf=None):  # 将偏置初始化到 Detect() 中，cf 是类别频率
        """初始化 Detect() 模块中 objectness 和类别的偏置项；可选地使用类别频率 `cf`。"""
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() 模块
        for mi, s in zip(m.m, m.stride):  # 遍历每层
            b = mi.bias.view(m.na, -1)  # 将卷积偏置(255) 转换为 (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (每幅640图片中有8个物体)
            b.data[:, 5: 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # 保留 YOLOv3 的 'Model' 类以保持向后兼容性

class SegmentationModel(DetectionModel):
    """实现基于 YOLOv3 的分割模型，具有可定制的配置、通道、类别和锚点。"""
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """通过可选的配置、通道数、类别数和锚点参数初始化分割模型。"""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """实现基于 YOLOv3 的图像分类模型，具有可配置的架构和类别数量。"""
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml 文件, 模型, 类别数量, 截断索引
        """通过检测模型或 YAML 文件初始化分类模型，支持可配置的类别数和截断索引。"""
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """通过 YOLOv3 检测模型初始化分类模型，配置类别和截断层。"""
        if isinstance(model, DetectMultiBackend):
            model = model.model  # 解包 DetectMultiBackend
        model.model = model.model[:cutoff]  # 截取 backbone 部分
        m = model.model[-1]  # 最后一层
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # 获取输入通道数
        c = Classify(ch, nc)  # 创建 Classify 层
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # 设置索引、来源、类型
        model.model[-1] = c  # 替换最后一层为分类层
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """从 YAML 文件配置创建 YOLOv3 分类模型。"""
        self.model = None

def parse_model(d, ch):  # model_dict, input_channels(3)
    """从字典中解析 YOLOv3 模型配置，并构建模型。"""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"], d.get("activation")
    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数, 如 Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # 输出激活函数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚点数量
    no = na * (nc + 5)  # 输出数量 = 锚点数 * (类别数 + 5)

    layers, save, c2 = [], [], ch[-1]  # 模型层，保存列表，输出通道数
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # 来自，数量，模块，参数
        m = eval(m) if isinstance(m, str) else m  # 评估字符串
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # 评估字符串

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 深度增益
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d,
            Focus, CrossConv, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d,
            DWConvTranspose2d, C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是输出层
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # 重复次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚点数量
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 附加索引，'from' 索引，类型，参数数量
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # 打印信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # 检查 YAML 文件
    print_args(vars(opt))
    device = select_device(opt.device)

    # 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)  # 随机生成输入图像
    model = Model(opt.cfg).to(device)  # 实例化模型

    # 选项
    if opt.line_profile:  # 逐层分析模型
        model(im, profile=True)

    elif opt.profile:  # 前向和后向传播分析
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # 测试所有 yolo*.yaml 配置
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)  # 实例化每个配置文件对应的模型
            except Exception as e:
                print(f"Error in {cfg}: {e}")  # 捕获并报告错误

    else:  # 输出融合后的模型摘要
        model.fuse()  # 将模型融合
