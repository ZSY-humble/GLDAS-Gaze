# -*- coding: utf-8 -*-
"""
@Author  : zsy
@Time    : 2025/5/19 下午2:59
@File    : position_encoding.py
@Desc    :
"""

import torch
import torch.nn as nn
from torchvision import models
import math
from os.path import join, dirname

# 类型注解模块（Python 3.5+特性，用于增强代码可读性）
# Optional[T] 表示参数可以是类型T或None
# 示例：def foo(x: Optional[int]) -> int: ...
from typing import Optional

import numpy as np
from typing import Tuple

'''
在处理图像、视觉注意力、眼动预测等任务中，神经网络模型需要知道每个“像素”或者“位置”在空间上的坐标信息。
而这些信息并不天然包含在图像向量中，所以我们人为添加“位置编码”，就像给每个位置贴上 GPS 地址一样，让模型能区分“这里”和“那里”。

✅ 1. PositionalEncoding（基础的一维位置编码）
✨ 作用：
给 Transformer 输入的token 序列添加位置编码，让模型具备顺序感（原始 Transformer 用的方式）。
📦 用法：
    用于 自然语言处理（NLP） 或其他一维序列输入（如词向量序列）。
    编码方式是固定的正余弦函数：
🎯 特点：
    固定、可解释、无参数；
    每个 token 位置都有唯一的编码；
    只支持一维序列。
    
✅ 2. PositionalEncoding2D（二维正余弦位置编码）
✨ 作用：
给图像或类似图像的二维结构（如 gaze/pointer 的 patch 序列）编码位置，分别对 X 和 Y 坐标使用正余弦函数编码，并拼接成一个向量。
📦 用法：
    用于输入是 图像块、眼动数据网格、patch grid 的任务；
    对每个坐标 (x, y) 生成编码
🎯 特点：
    与 PositionalEncoding 相同是 无参数、正余弦构造；
    支持 二维网格结构；
    编码是规则性的（周期性），存在“模糊性/折叠性”问题。
    
✅ 3. PositionEmbeddingRandom（随机傅里叶位置编码）
✨ 作用：
使用随机高斯频率矩阵将二维坐标 (x, y) 投影为嵌入空间，用于构建具有更高表达能力的非周期位置编码。
📦 用法：
    同样用于二维任务（图像块、gaze grid 等）；
    但相比 PositionalEncoding2D，它用随机的非固定方式编码位置，更适合处理复杂空间关系。
🎯 特点：
    使用随机初始化的矩阵，模型能力更强；
    与 PositionalEncoding2D 相比，不再是规则的周期编码，而是近似高斯傅里叶特征；
    更灵活，但编码不可解释、不确定。
    | 类名                        | 适用数据                 | 编码维度     | 编码方式    | 是否学习参数   | 特点                    |
| ------------------------- | -------------------- | -------- | ------- | -------- | --------------------- |
| `PositionalEncoding`      | 1D token 序列（如文本）     | 1D       | 正余弦     | 否        | 经典 transformer，简单有效   |
| `PositionalEncoding2D`    | 图像块、gaze patch、2D 网格 | 2D (x+y) | 正余弦     | 否        | 二维扩展版本，有周期性限制         |
| `PositionEmbeddingRandom` | 图像块、gaze patch、2D 网格 | 2D (x+y) | 随机傅里叶投影 | 否（但有随机性） | 更复杂、表达力强，适合处理空间相关复杂任务 |

'''

"""
这个类为一个二维空间（如图像）中的每个位置生成固定的正余弦位置编码。类似 Transformer 里给序列添加顺序信息，只不过这里是2D 的行列坐标。
"""
class PositionalEncoding2D(nn.Module):
    def __init__(self, pa, d_model, dropout, height=20, width=32, patch_num=None):
        super(PositionalEncoding2D, self).__init__()

        # 把总位置编码维度分成两半；一半用于行 (y)，一半用于列 (x)；
        # 这样可以将二维坐标信息分别编码后拼接，构成完整的位置表示。
        d_model = d_model // 2

        # 构造位置编码中使用的频率项
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)

        self.pa = pa
        self.n_special_symbols = len(pa.special_symbols)
        self.d_model = d_model

        # 构造行方向（Y轴）的正余弦编码
        pos_h = torch.arange(0, height).reshape(height, 1)  # 行索引
        pos_h_embedding = torch.zeros((height, d_model))
        pos_h_embedding[:, 0::2] = torch.sin(pos_h * den)
        pos_h_embedding[:, 1::2] = torch.cos(pos_h * den)

        # 构造列方向（X轴）的正余弦编码
        pos_w = torch.arange(0, width).reshape(width, 1)  # 列索引
        pos_w_embedding = torch.zeros((width, d_model))
        pos_w_embedding[:, 0::2] = torch.sin(pos_w * den)
        pos_w_embedding[:, 1::2] = torch.cos(pos_w * den)

        self.height = height
        self.width = width
        self.dropout = nn.Dropout(dropout)

        # 注册为 buffer，模型保存时一并保存，但不作为训练参数
        # 👉 类比：这是你“事先算好写在本子上的波形”，模型用它，但不会去改它。
        self.register_buffer('pos_w_embedding', pos_w_embedding)
        self.register_buffer('pos_h_embedding', pos_h_embedding)

    """
    forward：添加位置编码到序列（如眼动轨迹）
    根据 tgt_seq（目标序列）中的 gaze 行为位置，生成每个位置对应的 二维位置编码向量，并返回一个形状为 [batch, seq_len, d_model*2] 的编码张量。
    tgt_seq 是模型的输入序列（可能是 gaze 行为编码后的序列）。
    特殊符号（如 <PAD>、<EOS>）不参与 gaze 编码。
    每个 gaze 编码对应一个二维位置，需要解码为 (x, y) 坐标。
    """

    def forward(self, tgt_seq, scale=1):
        # 找出所有非 PAD 且非 EOS 的位置，也就是有效的 gaze 位置
        gaze_symbol_idx = torch.logical_and(tgt_seq != self.pa.pad_idx,
                                            tgt_seq != self.pa.eos_idx)

        # 初始化位置编码张量，形状为 [batch_size, seq_len, d_model*2]（因为 x 和 y 各占一半维度）
        pe = torch.zeros(*tgt_seq.shape, self.d_model * 2).to(tgt_seq.device)

        # 如果输入序列中没有任何 gaze 行为（全是 <PAD> 或 <EOS>），直接返回全零的编码
        if gaze_symbol_idx.sum() == 0:
            return pe

        # 从有效 gaze 编码中减去特殊符号数量，得到其实际在空间中的位置编号（action ID）
        actions = tgt_seq[gaze_symbol_idx] - self.n_special_symbols

        # 这两行代码就是把 gaze 的线性编号（从左到右、从上到下）还原成图像上的二维坐标 (y, x)，并且考虑了 patch 缩放（scale） 和 居中偏移。
        y = actions // (self.width / scale) + scale // 2
        x = actions % (self.width / scale) + scale // 2

        # 根据 (x, y) 坐标生成位置编码，拼接 x 和 y 的编码向量：[pe_x | pe_y]
        pe_valid = self.forward_pos(x, y)

        # 将编码结果填回原始位置张量中，非 gaze 行为的位置保持为全零
        pe[gaze_symbol_idx] = pe_valid

        # 返回完整的 [batch_size, seq_len, d_model*2] 的位置编码张量
        return pe

    """
    根据输入的二维坐标 (x, y)，查表获取其在 X 和 Y 方向上的位置编码，然后拼接成最终的二维位置编码向量。
    """

    def forward_pos(self, x, y):
        # 断言：x 和 y 坐标值不能超出预设的位置编码尺寸范围，否则报错
        assert x.max() < self.width and y.max() < self.height, "out of range"
        # 从列方向的位置编码表中，查出对应 x 坐标的编码向量（取出 pos_w_embedding 中索引为 x 的向量）
        pe_x = self.pos_w_embedding[x.long()]  # 形状：[batch_size, d_model]
        # 从行方向的位置编码表中，查出对应 y 坐标的编码向量（取出 pos_h_embedding 中索引为 y 的向量）
        pe_y = self.pos_h_embedding[y.long()]  # 形状：[batch_size, d_model]
        # 将 x 和 y 的编码向量拼接起来，形成完整的二维位置编码
        # 拼接后形状：[batch_size, d_model * 2]
        pe = torch.cat([pe_x, pe_y], dim=1)
        return pe  # 返回最终的二维位置编码


    def forward_featmaps(self, size, scale=1):
        """
        假设你输入的图像是 20x32 像素（高度 × 宽度），你要生成 1 个 10x16 特征图的位置编码（scale = 2）：
        scale用于生成空间位置编码的时候的“稀疏采样步长”：
        对这些采样点从已有的位置编码中抽取编码值；
        将 X、Y 编码分别扩展成 [d_model, h, w]，再拼接，得到 [2*d_model, h, w]；
        为了生成与特征图尺寸匹配的 2D 位置编码张量，最终形状是：[1, 2 * d_model, h, w] 用于给 Transformer 或 CNN 特征图提供位置信息。
        """
        h, w = size    # 解包特征图尺寸 (h: 高度, w: 宽度)

        # 检查尺寸是否和缩放后的位置编码尺寸一致
        # 例如：原图 height=20，scale=2，要求 h=10（20 / 2）
        assert h == math.ceil(self.height / scale) and w == math.ceil(
            self.width / scale), "wrong input"

        # 构建采样位置索引，起点是 scale//2，步长为 scale
        # 例如 scale=2，则采样位置为 1, 3, 5, ...  、而是每隔 scale 像素取一个位置编码，用来匹配对应 stride 的特征图。
        smp_ind_x = torch.arange(scale // 2, self.width, scale)  # 列方向采样点
        smp_ind_y = torch.arange(scale // 2, self.height, scale)  # 行方向采样点

        # 提取列方向（X轴）的位置编码，并转置 shape: [d_model, num_x]
        pe_x = self.pos_w_embedding[smp_ind_x].transpose(0, 1)

        # 提取行方向（Y轴）的位置编码，并转置 shape: [d_model, num_y]
        pe_y = self.pos_h_embedding[smp_ind_y].transpose(0, 1)

        # 扩展 pe_x：在中间插入维度并重复，使其 shape 变为 [d_model, h, w]
        pe_x = pe_x.unsqueeze(1).repeat(1, h, 1)

        # 扩展 pe_y：在末尾插入维度并重复，使其 shape 变为 [d_model, h, w]
        pe_y = pe_y.unsqueeze(2).repeat(1, 1, w)

        # 将 pe_x 和 pe_y 在“通道维度”上拼接，shape: [2 * d_model, h, w]
        pe = torch.cat([pe_x, pe_y], dim=0)

        # 增加 batch 维度，最终 shape: [1, 2 * d_model, h, w]
        # 位置编码只依赖于空间位置，不依赖于图像内容
        # # 所以批次中的所有图像共享相同的位置编码
        return pe.unsqueeze(0)

class PositionalEncoding(nn.Module):

    """
    这是一个用于添加位置编码（Positional Encoding）的辅助模块，
    它用于将 token（如词或patch）的位置信息注入 embedding 中，使模型具备“顺序感”。
    想象你有一句话：
    "I love deep learning!"
    每个单词先经过 embedding，但 Transformer 不知道哪是第一个单词哪是最后一个，于是我们用这个模块：
    给 I 加位置 0 的编码；
    给 love 加上位置 1 的编码；
    """

    def __init__(self, emb_size: int, maxlen: int = 100):
        """
        初始化位置编码。
        参数：
        - emb_size：每个 token 的 embedding 维度。
        - maxlen：支持的最大序列长度，决定预计算的位置编码行数。
        """
        super(PositionalEncoding, self).__init__()

        # 计算位置编码中不同频率的“分母项”
        # 频率按 log 等比下降，emb_size//2 个频率，对应偶数位置
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)

        # pos 是每个 token 在序列中的位置（0 到 maxlen-1）
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        # 初始化位置编码矩阵，形状为 (maxlen, emb_size)
        pos_embedding = torch.zeros((maxlen, emb_size))

        # 对于 embedding 的偶数位置，使用 sin(position * frequency)
        pos_embedding[:, 0::2] = torch.sin(pos * den)

        # 对于 embedding 的奇数位置，使用 cos(position * frequency)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # 添加一个维度，变成 (maxlen, 1, emb_size)，方便后续 broadcast 到 batch 中
        pos_embedding = pos_embedding.unsqueeze(-2)

        # register_buffer 表示 pos_embedding 是模型的一部分，但不是参数（不会被优化器更新）
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        """
        将预先计算好位置编码添加到 token embedding 上。
        参数：
        - token_embedding：形状为 (seq_len, batch_size, emb_size)

        返回：
        - 与 token_embedding 相同 shape 的位置编码部分
        """
        # 只取前 token_embedding.size(0) 个位置编码，形状为 (seq_len, 1, emb_size)
        return self.pos_embedding[:token_embedding.size(0), :]

    def forward_pos(self, pos: torch.Tensor):
        """
        按照指定位置索引取出对应的位置编码（通常用于手动构造位置）。
        参数：
        - pos：位置索引，如 tensor([1,3,7])，形状为 (N,)

        返回：
        - 对应的嵌入向量，形状为 (N, emb_size)
        """
        return self.pos_embedding[pos].squeeze(1)


class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率（随机傅里叶特征）生成的位置编码方式。
    本类使用的编码方式是 随机频率 + 正余弦变换（傅里叶特征），可以视为“从多个方向投影空间位置”的方法。
    它避免了传统规则正余弦位置编码的周期限制，更适合处理图像类的任意空间坐标。
    参考论文：https://arxiv.org/abs/2006.10739
    """

    def __init__(self, pa,
                 d_model: int,
                 dropout: float,
                 height: int = 20,
                 width: int = 32,
                 scale: Optional[float] = None, ) -> None:
        super().__init__()

        # 如果没指定 scale，默认使用 1.0
        if scale is None or scale <= 0.0:
            scale = 1.0

        self.pa = pa  # pa 用于记录特殊符号等信息
        self.n_special_symbols = len(pa.special_symbols)

        # Dropout 用于在训练时进行正则化
        self.dropout = nn.Dropout(dropout)

        # 网格高宽（决定位置坐标的最大值）
        self.height = height
        self.width = width

        # 将模型维度平均分配到 x 和 y（各一半）
        self.d_model = d_model // 2

        # 注册一个 buffer，不会作为模型参数参与训练，但会随模型保存
        # 生成一个 2 × (d_model/2) 的高斯随机矩阵，用于编码 x 和 y 坐标
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, self.d_model)),
        )

    def forward_pos(self, x: torch.Tensor, y: torch.Tensor, normalize=True) -> torch.Tensor:
        """
        对任意给定的 (x, y) 坐标生成随机傅里叶位置编码。

        参数：
        - x, y: 输入坐标张量
        - normalize: 是否将 x, y 归一化到 [0, 1] 区间（默认是）

        返回：
        - pe: 对应坐标的嵌入向量，形状为 (..., d_model*2)
        """
        if normalize:
            x, y = x / self.width, y / self.height  # 将坐标缩放到 [0, 1]

        coords = torch.stack([x, y], dim=-1)  # (..., 2)，将 x 和 y 堆叠为向量

        coords = 2 * coords - 1  # 缩放到 [-1, 1]，更适合后续嵌入

        # 与随机频率矩阵做线性投影： (..., 2) × (2, d_model) -> (..., d_model)
        coords = coords @ self.positional_encoding_gaussian_matrix

        coords = 2 * np.pi * coords  # 乘以 2π 进入周期空间（准备做 sin/cos）

        # 生成正余弦特征，拼接为最终位置编码：[..., d_model*2]
        pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return pe

    def forward_featmaps(self, size: Tuple[int, int], scale: int = 1) -> torch.Tensor:
        """
        为特征图生成位置编码。

        参数：
        - size: 输出特征图大小 (height, width)
        - scale: 缩放倍数

        返回：
        - pe: 位置编码张量，形状为 (C, H, W)
        """
        h, w = size  # 输出大小
        device = self.positional_encoding_gaussian_matrix.device

        # 创建一个全为 1 的占位网格
        grid = torch.ones((h, w), device=device, dtype=torch.float32)

        # 构造 x/y 网格坐标，每个点减 0.5 使其居中
        y_embed = grid.cumsum(dim=0) - 0.5  # 行方向
        x_embed = grid.cumsum(dim=1) - 0.5  # 列方向

        # 归一化到 [0, 1] 区间
        y_embed = y_embed / h
        x_embed = x_embed / w

        # 基于坐标生成位置编码：形状为 (H, W, C)
        pe = self.forward_pos(x_embed, y_embed, normalize=False)

        # 调整维度顺序为 (C, H, W)，以匹配 CNN 格式
        return pe.permute(2, 0, 1)

    def forward(self, tgt_seq, scale=1) -> torch.Tensor:
        """
        给定目标序列，生成每个位置的嵌入向量。

        参数：
        - tgt_seq: 目标动作序列（编号）
        - scale: 缩放系数，通常是 patch 大小

        返回：
        - pe: 每个动作对应的位置编码，形状为 (batch_size, d_model*2)
        """

        # 找出不是特殊符号（pad/eos）的有效 token 位置
        gaze_symbol_idx = torch.logical_and(tgt_seq != self.pa.pad_idx,
                                            tgt_seq != self.pa.eos_idx)

        # 初始化空的嵌入张量，全部为 0（形状为原序列 × d_model*2）
        pe = torch.zeros(*tgt_seq.shape, self.d_model * 2).to(tgt_seq.device)

        # 如果所有 token 都是特殊符号，直接返回零向量
        if gaze_symbol_idx.sum() == 0:
            return pe

        # 去除特殊符号编号偏移，映射回 patch ID
        actions = tgt_seq[gaze_symbol_idx] - self.n_special_symbols

        # 解码出 Y 坐标（在 patch 网格中）：除以每行的 patch 数再加上偏移
        y = actions // (self.width / scale) + scale // 2

        # 解码出 X 坐标：模除得到列索引，加上偏移
        x = actions % (self.width / scale) + scale // 2

        # 基于坐标生成位置编码
        pe_valid = self.forward_pos(x, y)

        # 把有效位置的编码赋值回 pe
        pe[gaze_symbol_idx] = pe_valid

        return pe

