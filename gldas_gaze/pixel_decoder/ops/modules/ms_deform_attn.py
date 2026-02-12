# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


# 检查一个数是否是2的幂次
def _is_power_of_2(n):
    # 类型和值检查：必须是非负整数
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    # 位运算技巧：n & (n-1) == 0 当且仅当 n 是2的幂
    # 例如：8(1000) & 7(0111) = 0，但9(1001) & 8(1000) = 8 ≠ 0
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        多尺度可变形注意力模块 - Deformable DETR的核心组件

        :param d_model      hidden dimension - 隐藏层维度（特征维度）
        :param n_levels     number of feature levels - 特征层级数（如res3,res4,res5=3级）
        :param n_heads      number of attention heads - 注意力头数（并行处理）
        :param n_points     number of sampling points per attention head per feature level
                           每个注意力头在每个特征层级上的采样点数
        """
        super().__init__()

        # 检查d_model是否能被n_heads整除（多头注意力的要求）
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))

        # 计算每个注意力头的维度
        _d_per_head = d_model // n_heads  # 例如：256 / 8 = 32

        # 检查每个头的维度是否是2的幂（CUDA实现更高效）
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        # im2col_step用于CUDA kernel的优化，控制并行处理的批次大小
        self.im2col_step = 128

        # 保存配置参数
        self.d_model = d_model  # 256
        self.n_levels = n_levels  # 3 (res3, res4, res5)
        self.n_heads = n_heads  # 8
        self.n_points = n_points  # 4

        # 采样偏移预测网络
        # 输入：d_model维的query特征
        # 输出：每个头、每个层级、每个采样点的2D偏移量(x,y)
        # 输出维度：n_heads * n_levels * n_points * 2 = 8 * 3 * 4 * 2 = 192
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # 注意力权重预测网络
        # 输出：每个头、每个层级、每个采样点的权重
        # 输出维度：n_heads * n_levels * n_points = 8 * 3 * 4 = 96
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # value投影层：将输入特征投影到value空间
        # d_model -> d_model (256 -> 256)
        self.value_proj = nn.Linear(d_model, d_model)

        # 输出投影层：将多头注意力结果投影回原始空间
        # d_model -> d_model (256 -> 256)
        self.output_proj = nn.Linear(d_model, d_model)

        # 初始化所有参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 1. 采样偏移的权重初始化为0
        constant_(self.sampling_offsets.weight.data, 0.)  # 维度：192 * 256

        # 2. 初始化采样偏移的偏置项（bias）- 创建一个圆形模式
        # 为每个注意力头生成一个角度
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # 例如8个头：[0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]

        # 将角度转换为单位圆上的点 (cos, sin)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # [[1.0, 0.0],    # 0°:   东
        #  [0.707, 0.707], # 45°:  东北
        #  [0.0, 1.0],     # 90°:  北
        #  [-0.707, 0.707],# 135°: 西北  类推

        # 归一化到[-1, 1]范围，并复制到所有层级和采样点
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)  #[8, 1, 1, 2]-》[8, 3, 4, 2]
        # 为不同的采样点设置不同的半径（距离）
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1  # 第i个点的半径为i+1

        # 设置为偏置项参数（这样不同的头会关注不同的方向）
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # 3. 注意力权重初始化为0
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        # 4. value投影使用Xavier初始化
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        # 5. 输出投影使用Xavier初始化
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        前向传播函数

        :param query                       (N, total_pixels, C)
                                          查询特征，通常来自transformer的输出
                                          N: batch size
                                          Length_{query}: 查询序列长度（所有层级的像素总数）
                                          C: 特征维度 (d_model=256)

        :param reference_points            (N, Length_{query}, n_levels, 2)
                                          参考点坐标，范围[0,1]，左上角(0,0)，右下角(1,1)
                                          或 (N, Length_{query}, n_levels, 4) 包含额外的宽高信息

        :param input_flatten               (N, total_pixels, C)
                                          展平的多尺度输入特征（所有层级拼接）

        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                                          每个层级的空间形状

        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
                                          每个层级在展平序列中的起始索引

        :param input_padding_mask          (N, sum of H_l*W_l over levels)
                                          padding掩码，True表示padding，False表示有效区域

        :return output                     (N, Length_{query}, C)
                                          输出特征
        """
        # 获取输入维度
        N, Len_q, _ = query.shape  # batch_size, 查询长度, 特征维度
        N, Len_in, _ = input_flatten.shape  # batch_size, 输入序列长度, 特征维度

        # 验证输入维度的一致性
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 1. 将输入特征投影到value空间
        value = self.value_proj(input_flatten)  # [N, Len_in, C]

        # 如果有padding掩码，将padding位置的value设为0
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # 重塑value以分离多头：[N, Len_in, n_heads, C/n_heads]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # 2. 预测采样偏移量
        # [N, Len_q, C] -> [N, Len_q, n_heads * n_levels * n_points * 2]
        sampling_offsets = self.sampling_offsets(query)
        # 重塑为：[N, Len_q, n_heads, n_levels, n_points, 2]
        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        # 3. 预测注意力权重
        # [N, Len_q, C] -> [N, Len_q, n_heads * n_levels * n_points]
        attention_weights = self.attention_weights(query)
        # 重塑并在最后一维做softmax归一化
        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1)  # 归一化权重
        # 最终形状：[N, Len_q, n_heads, n_levels, n_points]
        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # 4. 计算实际的采样位置
        # reference_points: [N, Len_q, n_levels, 2] - 参考点坐标
        if reference_points.shape[-1] == 2:
            # 标准情况：只有坐标信息
            # offset_normalizer用于将偏移量归一化到[0,1]范围
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # 采样位置 = 参考点 + 归一化的偏移量
            # 扩展维度以匹配：[N, Len_q, 1, n_levels, 1, 2] + [N, Len_q, n_heads, n_levels, n_points, 2]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # 包含边界框信息：(x, y, w, h)
            # 采样位置 = 参考点中心 + 偏移量 * 边界框大小的一半 / 采样点数
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # 5. 执行可变形注意力计算（使用 PyTorch 实现）
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        # output shape: [N, Len_q, C]
        # 6. 输出投影
        output = self.output_proj(output)

        return output
