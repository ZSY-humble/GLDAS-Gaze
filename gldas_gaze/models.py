from typing import Optional, Tuple, Dict, List
from torch import Tensor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import common.position_encoding as pe
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
import fvcore.nn.weight_init as weight_init
from .config import add_maskformer2_config
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

class StreamGatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, 1), nn.Sigmoid()
        )
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, q_dorsal: Tensor, q_ventral: Tensor) -> Tensor:
        g = self.gate(torch.cat([q_dorsal, q_ventral], dim=-1))
        return g * q_ventral + (1.0 - g) * q_dorsal

class CrossStreamInteraction(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float = 0.0, cross_stream_scale: float = 0.35):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_stream_scale = cross_stream_scale
        self.dorsal_to_ventral = CrossAttentionLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, normalize_before=False)
        self.ventral_to_dorsal = CrossAttentionLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, normalize_before=False)

    def forward(
        self, dorsal_embs: torch.Tensor, ventral_embs: torch.Tensor,
        dorsal_pos: Optional[torch.Tensor] = None, ventral_pos: Optional[torch.Tensor] = None,
        dorsal_padding_mask: Optional[torch.Tensor] = None,
        ventral_padding_mask: Optional[torch.Tensor] = None,
        extra_ventral: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        dorsal_enhanced, dorsal_attn_weights = self.ventral_to_dorsal(
            tgt=dorsal_embs, memory=ventral_embs, memory_key_padding_mask=ventral_padding_mask,
            pos=ventral_pos, query_pos=dorsal_pos, extra=extra_ventral,
        )
        ventral_enhanced, ventral_attn_weights = self.dorsal_to_ventral(
            tgt=ventral_embs, memory=dorsal_embs, memory_key_padding_mask=dorsal_padding_mask,
            pos=dorsal_pos, query_pos=ventral_pos, extra=None,
        )
        final_dorsal = dorsal_embs + self.cross_stream_scale * dorsal_enhanced
        final_ventral = ventral_embs + self.cross_stream_scale * ventral_enhanced
        return final_dorsal, final_ventral, {
            'dorsal_attn_weights': dorsal_attn_weights,
            'ventral_attn_weights': ventral_attn_weights,
        }

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
                     extra: Optional[Dict] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        tgt2, attn_weights = self.multihead_attn(
            query=q, key=k, value=v,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, attn_weights

    def forward_pre(self, tgt, memory, memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
                    extra: Optional[Dict] = None):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        tgt2 = self.multihead_attn(query=q, key=k, value=v,
                                   attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
                extra: Optional[Dict] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos, extra)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos, extra)

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0, activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class ImageFeatureEncoder(nn.Module):
    def __init__(self, cfg_path, dropout, pixel_decoder='MSD', pred_saliency=False):
        super(ImageFeatureEncoder, self).__init__()
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.merge_from_file(cfg_path)
        self.backbone = build_backbone(cfg)
        weights_path = cfg.MODEL.WEIGHTS
        try:
            bb_weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        except RuntimeError as e:
            if "zip" in str(e).lower() or "central directory" in str(e).lower():
                with open(weights_path, 'rb') as f:
                    bb_weights = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
            else:
                raise
        bb_weights_new = bb_weights.copy()
        for k, v in bb_weights.items():
            if k[:3] == 'res':
                bb_weights_new["stages." + k] = v
                bb_weights_new.pop(k)
        self.backbone.load_state_dict(bb_weights)
        self.backbone.eval()
        print('Loaded backbone weights from {}'.format(cfg.MODEL.WEIGHTS))

        if pred_saliency:
            self.saliency_head = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0)
            )
        else:
            self.saliency_head = None

        if cfg.MODEL.BACKBONE.NAME == 'D2SwinTransformer':
            input_shape = {
                "res2": ShapeSpec(channels=128, stride=4),
                "res3": ShapeSpec(channels=256, stride=8),
                "res4": ShapeSpec(channels=512, stride=16),
                "res5": ShapeSpec(channels=1024, stride=32)
            }
        else:
            input_shape = {
                "res2": ShapeSpec(channels=256, stride=4),
                "res3": ShapeSpec(channels=512, stride=8),
                "res4": ShapeSpec(channels=1024, stride=16),
                "res5": ShapeSpec(channels=2048, stride=32)
            }

        args = {
            'input_shape': input_shape, 'conv_dim': 256, 'mask_dim': 256, 'norm': 'GN',
            'transformer_dropout': dropout, 'transformer_nheads': 8,
            'transformer_dim_feedforward': 1024, 'transformer_enc_layers': 6,
            'transformer_in_features': ['res3', 'res4', 'res5'], 'common_stride': 4,
        }
        self.ema_256 = EMA(256)
        self.ema_512 = EMA(512)
        self.ema_1024 = EMA(1024)
        self.ema_2048 = EMA(2048)

        if pixel_decoder == 'MSD':
            msd = MSDeformAttnPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_MSDeformAttnPixelDecoder.pkl'
            msd_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
            msd_weights_new = msd_weights.copy()
            for k, v in msd_weights.items():
                if k[:7] == 'adapter':
                    msd_weights_new["lateral_convs." + k] = v
                    msd_weights_new.pop(k)
                elif k[:5] == 'layer':
                    msd_weights_new["output_convs." + k] = v
                    msd_weights_new.pop(k)
            msd.load_state_dict(msd_weights_new)
            print('Loaded MSD pixel decoder weights from {}'.format(ckpt_path))
            self.pixel_decoder = msd
            self.pixel_decoder.eval()
        else:
            raise NotImplementedError

    def forward(self, x):
        features = self.backbone(x)
        features['res2'] = self.ema_256(features['res2'])
        features['res3'] = self.ema_512(features['res3'])
        features['res4'] = self.ema_1024(features['res4'])
        features['res5'] = self.ema_2048(features['res5'])
        high_res_featmaps, _, ms_feats = self.pixel_decoder.forward_features(features)
        if self.saliency_head is not None:
            saliency_map = self.saliency_head(high_res_featmaps)
            return {'pred_saliency': saliency_map}
        else:
            return high_res_featmaps, ms_feats[0], ms_feats[1]

class Learnable2DPosEncoding(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
        coords = torch.stack([x, y], dim=-1).view(-1, 2)
        self.register_buffer("coords", coords)

    def forward(self, B: int):
        pos = self.proj(self.coords)
        pos = pos.unsqueeze(1).repeat(1, B, 1)
        return pos

class GLDASGazeModel(nn.Module):
    def __init__(
        self, pa, num_decoder_layers: int, hidden_dim: int, nhead: int, ntask: int,
        tgt_vocab_size: int, num_output_layers: int, separate_fix_arch: bool = False,
        train_encoder: bool = False, train_foveator: bool = True, train_pixel_decoder: bool = True,
        use_dino: bool = True, pre_norm: bool = False, dropout: float = 0.0,
        dim_feedforward: int = 512, parallel_arch: bool = True, dorsal_source: List[str] = ["P1"],
        num_encoder_layers: int = 3, output_centermap: bool = False, output_saliency: bool = False,
        output_target_map: bool = False, combine_pos_emb: bool = True, combine_all_emb: bool = False,
        transfer_learning_setting: str = 'none', project_queries: bool = True,
        is_pretraining: bool = False, output_feature_map_name: str = 'P2',
        cross_stream_scale: float = 0.35,
    ):
        super(GLDASGazeModel, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.is_pretraining = is_pretraining
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.output_feature_map_name = output_feature_map_name
        self.parallel_arch = parallel_arch
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout, pa.pixel_decoder)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.train_pixel_decoder = train_pixel_decoder
        if train_pixel_decoder:
            self.encoder.pixel_decoder.train()
            for p in self.encoder.pixel_decoder.parameters():
                p.requires_grad = True
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        self.transfer_learning_setting = transfer_learning_setting
        self.project_queries = project_queries
        self.ntask = ntask
        self.aux_queries = 0
        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.query_pos = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.cross_stream_interactions = nn.ModuleList([
            CrossStreamInteraction(hidden_dim, nhead, dropout, cross_stream_scale=cross_stream_scale) for _ in range(self.num_decoder_layers)
        ])
        self.query_dorsal_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, nhead, dropout, normalize_before=pre_norm) for _ in range(self.num_decoder_layers)
        ])
        self.query_ventral_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, nhead, dropout, normalize_before=pre_norm) for _ in range(self.num_decoder_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFNLayer(hidden_dim, dim_feedforward, dropout, normalize_before=pre_norm) for _ in range(self.num_decoder_layers)
        ])
        self.stream_fusions = nn.ModuleList([StreamGatedFusion(hidden_dim) for _ in range(self.num_decoder_layers)])
        self.num_encoder_layers = num_encoder_layers
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.termination_predictor = MLP(hidden_dim + 1, hidden_dim, 1, num_output_layers)
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim, num_output_layers)
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa, hidden_dim, height=pa.im_h // 4, width=pa.im_w // 4, dropout=dropout)
        if self.output_feature_map_name == 'P4':
            self.pos_scale = 1
        elif self.output_feature_map_name == 'P2':
            self.pos_scale = 4
        else:
            raise NotImplementedError
        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)
        self.query_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_decoder_layers)])

        self.learnable_pos_enc = Learnable2DPosEncoding(hidden_dim, height=pa.im_h // 4, width=pa.im_w // 4)

        if self.is_pretraining:
            self.pretrain_task_head = MLP(hidden_dim, 32, 1, 3)

    def forward(self, img, tgt_seq, tgt_padding_mask, tgt_seq_high, task_ids=None, return_attn_weights=False, img_ids=None):
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError

        bs, c, H, W = high_res_featmaps.shape

        dorsal_embs, dorsal_pos = [], []
        if "P1" in self.dorsal_source:
            img_embs = self.input_proj(img_embs_s1)
            _bs, _c, h, w = img_embs.shape
            pe2d = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=8)
            emb = img_embs.view(_bs, _c, -1).permute(2, 0, 1)
            dorsal_embs.append(emb)
            dorsal_pos.append(pe2d.expand(_bs, _c, h, w).view(_bs, _c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            img_embs = self.input_proj(img_embs_s2)
            _bs, _c, h, w = img_embs.shape
            pe2d = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=4)
            emb = img_embs.view(_bs, _c, -1).permute(2, 0, 1)
            dorsal_embs.append(emb)
            dorsal_pos.append(pe2d.expand(_bs, _c, h, w).view(_bs, _c, -1).permute(2, 0, 1))
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)

        tgt_seq_high = tgt_seq_high.transpose(0, 1)
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt_seq_high == 0).transpose(0, 1)
        
        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        ventral_embs = torch.gather(
            torch.cat([torch.zeros(1, *highres_embs.shape[1:], device=highres_embs.device), highres_embs], dim=0),
            0, tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape, highres_embs.size(-1))
        )
        ventral_pos = self.pixel_loc_emb(tgt_seq_high)

        grid_pos_full = self.learnable_pos_enc(bs)
        idx0 = torch.clamp(tgt_seq_high - 1, min=0)
        ventral_learnable_pos = grid_pos_full.gather(0, idx0.unsqueeze(-1).expand(-1, -1, grid_pos_full.size(-1)))

        if self.combine_pos_emb:
            dorsal_embs = dorsal_embs + dorsal_pos
            ventral_embs = ventral_embs + ventral_pos + 0.5 * ventral_learnable_pos
            dorsal_pos_attn = torch.zeros_like(dorsal_pos)
            ventral_pos_attn = torch.zeros_like(ventral_pos)
        else:
            dorsal_pos_attn = dorsal_pos
            ventral_pos_attn = ventral_pos + 0.5 * ventral_learnable_pos


        time_emb = self.fix_ind_emb.weight[:ventral_embs.size(0)].unsqueeze(1).expand_as(ventral_embs)
        ventral_embs = ventral_embs * (1 + time_emb)
        ventral_pos_attn = ventral_pos_attn.clone()
        ventral_pos_attn[tgt_padding_mask.transpose(0, 1)] = 0
        output_featmaps = output_featmaps + self.pixel_loc_emb.forward_featmaps(output_featmaps.shape[-2:], scale=self.pos_scale)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None
        num_fixs = (tgt_padding_mask.size(1) - torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0).expand(self.ntask, bs)
        attn_weights_all_layers = []
        current_dorsal = dorsal_embs
        current_ventral = ventral_embs

        for i in range(self.num_decoder_layers):
            query_residual = query_embed
            enhanced_dorsal, enhanced_ventral, _ = self.cross_stream_interactions[i](
                dorsal_embs=current_dorsal, ventral_embs=current_ventral,
                dorsal_pos=dorsal_pos_attn, ventral_pos=ventral_pos_attn,
                dorsal_padding_mask=None, ventral_padding_mask=tgt_padding_mask, extra_ventral=None,
            )
            q_dorsal, dorsal_attn_w = self.query_dorsal_cross_attn[i](
                tgt=query_embed, memory=enhanced_dorsal, memory_key_padding_mask=None,
                pos=dorsal_pos_attn, query_pos=query_pos, extra=None
            )
            q_ventral, ventral_attn_w = self.query_ventral_cross_attn[i](
                tgt=query_embed, memory=enhanced_ventral, memory_key_padding_mask=tgt_padding_mask,
                pos=ventral_pos_attn, query_pos=query_pos, extra=None
            )
            fused_q = self.stream_fusions[i](q_dorsal, q_ventral)
            query_embed = self.ffn_layers[i](fused_q)
            query_embed = self.query_norm[i](query_embed + query_residual)
            current_dorsal = enhanced_dorsal
            current_ventral = enhanced_ventral
            if return_attn_weights:
                attn_weights_all_layers.append(dorsal_attn_w)

        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        else:
            raise NotImplementedError

        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = self.termination_predictor(x)
        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed, output_featmaps)

        out = {"pred_termination": output_termination.squeeze(-1).transpose(0, 1)}

        if self.training:
            out["pred_fixation_map"] = outputs_fixation_map.transpose(0, 1)
        else:
            outputs_fixation_map = outputs_fixation_map[task_ids, torch.arange(bs)]
            outputs_fixation_map = torch.sigmoid(outputs_fixation_map)
            outputs_fixation_map = F.interpolate(outputs_fixation_map.unsqueeze(1), size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
            out["pred_fixation_map"] = outputs_fixation_map

        if self.training:
            ind = self.ntask
        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out

    def encode(self, img: torch.Tensor):
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError
        dorsal_embs, dorsal_pos = [], []
        if "P1" in self.dorsal_source:
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe2d = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=8)
            emb = img_embs.view(bs, c, -1).permute(2, 0, 1)
            dorsal_embs.append(emb)
            dorsal_pos.append(pe2d.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            img_embs = self.input_proj(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe2d = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:], scale=4)
            emb = img_embs.view(bs, c, -1).permute(2, 0, 1)
            dorsal_embs.append(emb)
            dorsal_pos.append(pe2d.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        return dorsal_embs, dorsal_pos, None, (high_res_featmaps, output_featmaps)

    def decode_and_predict(self, dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps,
                          tgt_seq, tgt_padding_mask, tgt_seq_high, task_ids=None, return_attn_weights=False):
        high_res_featmaps, output_featmaps = high_res_featmaps
        bs, c, H, W = high_res_featmaps.shape
        highres_seq = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        tgt_seq_high = tgt_seq_high.transpose(0, 1)
        if dorsal_mask is None:
            dorsal_mask = torch.zeros(1, *highres_seq.shape[1:], device=dorsal_embs.device)
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt_seq_high == 0).transpose(0, 1)
        
        ventral_embs = torch.gather(
            torch.cat([dorsal_mask, highres_seq], dim=0), 0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape, highres_seq.size(-1))
        )
        ventral_pos = self.pixel_loc_emb(tgt_seq_high)
        grid_pos_full = self.learnable_pos_enc(bs)
        idx0 = torch.clamp(tgt_seq_high - 1, min=0)
        ventral_learnable_pos = grid_pos_full.gather(0, idx0.unsqueeze(-1).expand(-1, -1, grid_pos_full.size(-1)))

        if self.combine_pos_emb:
            dorsal_embs = dorsal_embs + dorsal_pos
            ventral_embs = ventral_embs + ventral_pos + 0.5 * ventral_learnable_pos
            dorsal_pos_attn = torch.zeros_like(dorsal_pos)
            ventral_pos_attn = torch.zeros_like(ventral_pos)
        else:
            dorsal_pos_attn = dorsal_pos
            ventral_pos_attn = ventral_pos + 0.5 * ventral_learnable_pos

        time_emb = self.fix_ind_emb.weight[:ventral_embs.size(0)].unsqueeze(1).expand_as(ventral_embs)
        ventral_embs = ventral_embs * (1 + time_emb)
        ventral_pos_attn = ventral_pos_attn.clone()
        ventral_pos_attn[tgt_padding_mask.transpose(0, 1)] = 0
        output_featmaps_wpos = output_featmaps + self.pixel_loc_emb.forward_featmaps(output_featmaps.shape[-2:], self.pos_scale)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None
        num_fixs = torch.ones(self.ntask, bs).to(query_embed.device) * tgt_seq_high.size(0)
        attn_weights_all_layers = []
        current_dorsal = dorsal_embs
        current_ventral = ventral_embs

        for i in range(self.num_decoder_layers):
            query_residual = query_embed
            enhanced_dorsal, enhanced_ventral, _ = self.cross_stream_interactions[i](
                dorsal_embs=current_dorsal, ventral_embs=current_ventral,
                dorsal_pos=dorsal_pos_attn, ventral_pos=ventral_pos_attn,
                dorsal_padding_mask=None, ventral_padding_mask=tgt_padding_mask, extra_ventral=None,
            )
            q_dorsal, dorsal_attn_w = self.query_dorsal_cross_attn[i](
                tgt=query_embed, memory=enhanced_dorsal, memory_key_padding_mask=None,
                pos=dorsal_pos_attn, query_pos=query_pos, extra=None
            )
            q_ventral, ventral_attn_w = self.query_ventral_cross_attn[i](
                tgt=query_embed, memory=enhanced_ventral, memory_key_padding_mask=tgt_padding_mask,
                pos=ventral_pos_attn, query_pos=query_pos, extra=None
            )
            fused_q = self.stream_fusions[i](q_dorsal, q_ventral)
            query_embed = self.ffn_layers[i](fused_q)
            query_embed = self.query_norm[i](query_embed + query_residual)
            current_dorsal = enhanced_dorsal
            current_ventral = enhanced_ventral
            if return_attn_weights:
                attn_weights_all_layers.append(dorsal_attn_w)

        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        else:
            raise NotImplementedError

        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = torch.sigmoid(self.termination_predictor(x))
        out = {"pred_termination": output_termination.squeeze(-1)[task_ids, torch.arange(bs)]}
        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed, output_featmaps_wpos)
        outputs_fixation_map = torch.sigmoid(outputs_fixation_map[task_ids, torch.arange(bs)])
        outputs_fixation_map = F.interpolate(outputs_fixation_map.unsqueeze(1), size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
        out["pred_fixation_map"] = outputs_fixation_map.view(bs, -1)
        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out
