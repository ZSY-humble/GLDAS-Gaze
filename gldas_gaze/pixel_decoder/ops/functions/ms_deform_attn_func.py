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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError:
    MSDA = None


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        if MSDA is None:
            raise ModuleNotFoundError(
                "Please compile MultiScaleDeformableAttention CUDA op: "
                "cd gldas_gaze/pixel_decoder/ops && sh make.sh"
            )
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if MSDA is None:
            raise ModuleNotFoundError("MultiScaleDeformableAttention not available")
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    多尺度可变形注意力的核心计算
    功能：根据预测的采样位置和权重，从多尺度特征中采样并加权聚合

    这是PyTorch实现版本，实际使用时会用CUDA版本以提高效率
    """
    # 获取输入张量的维度
    N_, S_, M_, D_ = value.shape
    # N_: batch size (批次大小)
    # S_: 所有层级的总像素数，例如 H1*W1 + H2*W2 + H3*W3 = 10000 + 2500 + 625 = 13125
    # M_: 注意力头数，例如 8
    # D_: 每个头的维度，例如 256/8 = 32

    # 将拼接的value按照每个层级的像素数分割开
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # 将采样位置从[0,1]范围转换到[-1,1]范围（F.grid_sample的要求）
    sampling_grids = 2 * sampling_locations - 1

    # 初始化列表，用于存储每个层级的采样结果
    sampling_value_list = []

    # 获取采样位置张量的维度
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    # 遍历每个特征层级
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # 重塑当前层级的value，从序列格式转换为图像格式
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)

        # 获取当前层级的采样网格坐标
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)

        # 使用双线性插值进行采样
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)

        sampling_value_list.append(sampling_value_l_)

    # 重塑注意力权重，准备与采样值相乘
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)

    # 计算最终输出：堆叠所有层级的采样值，应用注意力权重，求和
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)

    # 转置输出并确保内存连续性
    return output.transpose(1, 2).contiguous()
