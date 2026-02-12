# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=3, enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model  # 特征维度，默认256
        self.nhead = nhead  # 多头注意力的头数，默认8

        # 创建编码器层，包含多尺度可变形注意力和FFN
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        # 创建编码器，由多个编码器层堆叠而成
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        # 可学习的层级嵌入，用于区分不同特征层级
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化所有参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 使用Xavier初始化多维参数
        # 特殊处理MSDeformAttn模块的参数初始化
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # 使用正态分布初始化层级嵌入
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        # 计算有效区域的比例，用于处理padding
        _, H, W = mask.shape  # mask shape: [batch_size, H, W]
        # 计算每行和每列的有效（非mask）像素数
        valid_H = torch.sum(~mask[:, :, 0], 1)  # 每个样本的有效高度
        valid_W = torch.sum(~mask[:, 0, :], 1)  # 每个样本的有效宽度
        # 计算有效比例
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [batch_size, 2]
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        # srcs: 多尺度特征图列表 [feat1, feat2, feat3, feat4]
        # pos_embeds: 对应的位置编码列表

        # 为每个特征图创建全零mask（表示所有位置都有效）
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        # 准备编码器输入
        src_flatten = []  # 展平的特征
        mask_flatten = []  # 展平的mask
        lvl_pos_embed_flatten = []  # 展平的位置编码（包含层级信息）
        spatial_shapes = []  # 记录每个层级的空间形状

        # 处理每个特征层级
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape  # [batch_size, channels, height, width]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # 将特征图展平并转置：[bs, c, h, w] -> [bs, h*w, c]
            src = src.flatten(2).transpose(1, 2)
            # 将mask展平：[bs, h, w] -> [bs, h*w]
            mask = mask.flatten(1)
            # 将位置编码展平并转置：[bs, c, h, w] -> [bs, h*w, c]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            # 添加层级嵌入到位置编码中，用于区分不同层级
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # 将所有层级的特征拼接在一起
        src_flatten = torch.cat(src_flatten, 1)  # [bs, total_pixels, c]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs, total_pixels]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [bs, total_pixels, c]

        # 将空间形状转换为tensor
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 计算每个层级在展平序列中的起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算每个层级的有效区域比例
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # 通过编码器处理
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()

        # 多尺度可变形自注意力模块
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, d_ffn)  # 第一个线性层，扩展维度
        self.activation = _get_activation_fn(activation)  # 激活函数
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)  # 第二个线性层，恢复维度
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 将位置编码加到特征上（如果有位置编码的话）
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # 前馈网络处理
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # src: 输入特征 [bs, total_pixels, d_model]
        # pos: 位置编码
        # reference_points: 参考点坐标
        # spatial_shapes: 各层级的空间形状
        # level_start_index: 各层级的起始索引
        # padding_mask: padding掩码

        # 自注意力模块
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        # src2 shape: [N, total_pixels, C]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 前馈网络
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 复制编码器层，创建多层编码器
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # 生成参考点，用于可变形注意力
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 为每个特征层级生成网格坐标
            # meshgrid生成H_×W_的网格，坐标从0.5开始（像素中心）
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 展平并归一化坐标
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [bs, H_*W_, 2]
            reference_points_list.append(ref)

        # 拼接所有层级的参考点
        reference_points = torch.cat(reference_points_list, 1)  # [bs, total_pixels, 2]
        # 扩展维度并应用有效区域比例
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [bs, total_pixels, n_levels, 2]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        # 生成参考点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        # 逐层处理
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],  # 例如: {'res2': ShapeSpec(channels=256, stride=4), 'res3': ShapeSpec(channels=512, stride=8), ...}
            *,
            transformer_dropout: float,  # 0.1 或其他值，用于防止过拟合
            transformer_nheads: int,  # 8个注意力头，并行处理不同的特征表示
            transformer_dim_feedforward: int,  # 1024维，FFN的中间层维度（先扩展到1024再压缩回256）
            transformer_enc_layers: int,  # 6层编码器，逐层细化特征
            conv_dim: int,  # 256维，所有特征统一投影到这个维度
            mask_dim: int,  # 256维，最终mask预测的特征维度
            norm: Optional[Union[str, Callable]] = None,  # 'GN'表示Group Normalization
            # deformable transformer encoder args
            transformer_in_features: List[str],  # ['res3', 'res4', 'res5'] - 只使用这三个层级的特征
            common_stride: int,  # 4，表示最终输出相对原图缩小4倍
    ):
        """
        配置示例解释：
        - input_shape: 包含所有ResNet特征层的信息（res2-res5）
        - transformer_in_features: ['res3', 'res4', 'res5'] 表示只将这三个层级送入Transformer
        - conv_dim=256: 所有特征都会被投影到256维，保证维度一致
        - common_stride=4: 最终输出是原图1/4大小，适合密集预测任务
        """
        super().__init__()

        # 筛选transformer使用的特征
        # 从完整的input_shape中，只保留transformer_in_features指定的特征层
        # 结果: {'res3': ShapeSpec(...), 'res4': ShapeSpec(...), 'res5': ShapeSpec(...)}
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # 按stride排序所有输入特征（包括res2）
        # 排序后: [('res2', stride=4), ('res3', stride=8), ('res4', stride=16), ('res5', stride=32)]
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # ["res2", "res3", "res4", "res5"]
        self.feature_strides = [v.stride for k, v in input_shape]  # [4, 8, 16, 32]
        self.feature_channels = [v.channels for k, v in input_shape]  # [256, 512, 1024, 2048] (典型的ResNet通道数)

        # 处理transformer输入特征（只有res3, res4, res5）
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # ["res3", "res4", "res5"]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]  # [512, 1024, 2048]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # [8, 16, 32]

        self.transformer_num_feature_levels = len(self.transformer_in_features)  # 3个层级

        # 输入投影层：将不同通道数的特征映射到统一的256维
        if self.transformer_num_feature_levels > 1:  # 我们有3个层级，所以进入这个分支
            input_proj_list = []
            # 注意这里反转了顺序，从res5到res3处理（从2048到512通道）
            for in_channels in transformer_in_channels[::-1]:  # [2048, 1024, 512]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),  # 1x1卷积: in_channels -> 256
                    nn.GroupNorm(32, conv_dim),  # 32组的组归一化，作用于256个通道
                ))
            self.input_proj = nn.ModuleList(input_proj_list)  # 3个投影层
        else:
            # 单层级情况（不会执行到）
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        # Xavier初始化投影层权重
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 创建Transformer编码器
        # 配置：256维特征，8头注意力，1024维FFN，6层编码器，3个特征层级
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,  # 256
            dropout=transformer_dropout,  # dropout率
            nhead=transformer_nheads,  # 8
            dim_feedforward=transformer_dim_feedforward,  # 1024
            num_encoder_layers=transformer_enc_layers,  # 6
            num_feature_levels=self.transformer_num_feature_levels,  # 3
        )

        # 位置编码生成器
        N_steps = conv_dim // 2  # 256 // 2 = 128
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim  # 256
        # 最终的mask特征生成层：256 -> 256（保持维度不变）
        self.mask_features = Conv2d(
            conv_dim,  # 256
            mask_dim,  # 256
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # 固定使用3个尺度输出
        self.common_stride = common_stride  # 4

        # 计算需要多少个FPN层级
        # min(transformer_feature_strides) = 8 (res3的stride)
        # log2(8) - log2(4) = 3 - 2 = 1
        # 需要1个额外的FPN层级来达到stride=4的分辨率
        stride = min(self.transformer_feature_strides)  # 8
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))  # 1

        # 创建FPN层（只需要1个，用于res2）
        lateral_convs, lc_names = [], []
        output_convs, oc_names = [], []

        use_bias = norm == ""  # 使用GN时不需要bias，所以use_bias=False
        # 只处理res2（因为num_fpn_levels=1）
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):  # [256]
            lateral_norm = get_norm(norm, conv_dim)  # GroupNorm
            output_norm = get_norm(norm, conv_dim)  # GroupNorm

            # 横向连接：256 -> 256（res2已经是256通道，所以只是加上归一化）
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            # 输出卷积：256 -> 256，3x3卷积用于特征融合
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            lc_names.append("adapter_{}".format(idx + 1))  # "adapter_1"
            oc_names.append("layer_{}".format(idx + 1))  # "layer_1"
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # 因为只有1个FPN层，反转后还是1个
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        # 转换为ModuleDict
        self.lateral_convs = nn.ModuleDict(zip(lc_names[::-1], self.lateral_convs))
        self.output_convs = nn.ModuleDict(zip(oc_names[::-1], self.output_convs))

    @autocast(enabled=False)
    def forward_features(self, features):
        """
        输入features包含: res2(stride=4), res3(stride=8), res4(stride=16), res5(stride=32)
        但只有res3, res4, res5会通过Transformer
        """
        srcs = []  # 存储投影后的特征
        pos = []  # 存储位置编码

        # 处理transformer特征：按照res5->res4->res3顺序
        for idx, f in enumerate(self.transformer_in_features[::-1]):  # ['res5', 'res4', 'res3']
            x = features[f].float()  # 获取对应特征，转为float32
            srcs.append(self.input_proj[idx](x))  # 投影到256维
            pos.append(self.pe_layer(x))  # 生成位置编码

        # 通过6层Transformer编码器处理3个尺度的特征
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        # 将transformer输出按层级分割
        split_size_or_sections = [None] * self.transformer_num_feature_levels  # 3个层级
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0

        # 将每个层级的特征重塑回2D形式
        # 此时out包含3个特征图：res5(stride=32), res4(stride=16), res3(stride=8)
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # FPN处理：只处理res2
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):  # ['res2']
            x = features[f].float()  # res2特征，stride=4
            lateral_conv = list(self.lateral_convs.values())[idx]  # adapter_1
            output_conv = list(self.output_convs.values())[idx]  # layer_1

            # 横向连接
            cur_fpn = lateral_conv(x)  # res2: 256 -> 256

            # 上采样res3(stride=8)到res2的分辨率(stride=4)并融合
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)  # 3x3卷积融合
            out.append(y)  # 添加stride=4的特征

        # 收集3个尺度的特征用于mask预测
        # out现在包含4个特征图，选择前3个：
        # out[0]: res5 (stride=32)
        # out[1]: res4 (stride=16)
        # out[2]: res3 (stride=8)
        # out[3]: res2融合后 (stride=4)
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:  # 收集3个
                multi_scale_features.append(o)
                num_cur_levels += 1

        # 返回：
        # 1. mask_features: 基于最高分辨率特征(stride=4)生成的mask特征
        # 2. out[0]: 最低分辨率特征(res5, stride=32)
        # 3. multi_scale_features: 3个尺度的特征列表
        return self.mask_features(out[-1]), out[0], multi_scale_features