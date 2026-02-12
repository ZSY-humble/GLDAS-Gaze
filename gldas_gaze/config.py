"""
CfgNode 是 Detectron2 中用来构建和管理配置参数的一个类
你可以把它理解为一个“可以嵌套的字典”，专门用来处理深度学习项目中的超参数配置，比如：
    模型结构（如层数、隐藏维度等）
    数据输入方式（如图像大小、强方式等）
    训练参数（如学习率、优化器类型、权重衰减等）
    测试策略（如阈值、是否启用某些模块等）
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    添加 MaskFormer2 模型所需的所有配置项
    """

    # ---------------------- 输入设置 ----------------------
    # 数据集处理模块的名称，决定使用哪种数据读取和预处理方式
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    # 是否使用 SSD 风格的数据增强（颜色扰动）
    cfg.INPUT.COLOR_AUG_SSD = False

    # 随机裁剪时，允许语义类别最大覆盖区域（1.0 表示不限制）
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0

    # 输入图像是否需要 padding 以满足尺寸可整除性（-1 表示不强制）
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # ---------------------- 优化器设置 ----------------------
    # 嵌入层的权重衰减（正则项）
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0

    # 使用哪种优化器，这里为 AdamW（适用于 Transformer）
    cfg.SOLVER.OPTIMIZER = "ADAMW"

    # 骨干网络的学习率缩放倍率
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # ---------------------- MaskFormer 模型配置 ----------------------
    cfg.MODEL.MASK_FORMER = CN()

    # ---------- 损失函数设置 ----------
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True  # 使用深监督（多层输出监督）
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1  # “无目标”类别的权重
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0  # 分类损失权重
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0  # Dice 损失权重
    cfg.MODEL.MASK_FORMER.NORMAL_WEIGHT = 1.0  # 法线估计损失权重
    cfg.MODEL.MASK_FORMER.DEPTH_WEIGHT = 1.0  # 深度估计损失权重
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0  # 掩膜损失权重（很高）

    # ---------- Transformer 设置 ----------
    cfg.MODEL.MASK_FORMER.NHEADS = 8  # 多头注意力头数
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1  # dropout 概率
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048  # FFN 层维度
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0  # encoder 层数（为0表示无 encoder）
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6  # decoder 层数
    cfg.MODEL.MASK_FORMER.PRE_NORM = False  # 是否使用 pre-norm 模式
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256  # Transformer 隐藏层维度
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100  # 用于实例分割的查询数量

    # 使用 backbone 的哪个特征图作为 Transformer 输入
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"

    # 是否强制输入 projection（当输入维度不同步时）
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # ---------- 推理阶段设置 ----------
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True  # 开启语义分割输出
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False  # 是否开启实例分割
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False  # 是否开启全景分割
    cfg.MODEL.MASK_FORMER.TEST.NORMAL_ON = False  # 是否开启法线输出
    cfg.MODEL.MASK_FORMER.TEST.DEPTH_ON = False  # 是否开启深度估计
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0  # 掩码置信度阈值
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0  # 区域重叠阈值
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.TEST.GT_MASK_NORMAL = False  # 是否使用 GT 掩码监督法线

    # 输入图像尺寸可整除性（如 32 表示输入尺寸需是 32 的倍数）
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # ---------------------- Pixel Decoder 设置 ----------------------
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256  # 掩码维度
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0  # Transformer 编码层数
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # ---------------------- Swin Transformer Backbone ----------------------
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # ---------------------- MaskFormer2 特有设置 ----------------------
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ 增强（大尺度随机缩放增强）
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # 多尺度可变形注意力（MSDeformAttn）配置
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # ---------------------- Point-based 掩码训练配置 ----------------------
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112  # 每次训练采样点数
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0  # 过采样比例
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75  # 重要性采样比例
    cfg.MODEL.MASK_FORMER.SEG_ON = True  # 开启语义分割
    cfg.MODEL.MASK_FORMER.USE_CONTRASTIVE_LOSS = True  # 对比损失
    cfg.MODEL.MASK_FORMER.USE_PER_IMG_LOSS = True  # 每图像损失
    cfg.MODEL.MASK_FORMER.USE_PER_SEG_LOSS = True  # 每分割块损失
    cfg.MODEL.MASK_FORMER.USE_INTERM_NORMAL_LOSS = False  # 是否使用中间层法线监督
    cfg.MODEL.MASK_FORMER.USE_MASK_ATTN_NORMAL = True  # 使用掩码注意力预测法线
    cfg.MODEL.MASK_FORMER.USE_SEGMENTAL_NORMAL = True  # 使用分割结构预测法线
    cfg.MODEL.MASK_FORMER.SEGMENTATION_AS_AUX = False  # 是否作为辅助分割任务

    # ---------------------- 深度估计头部配置 ----------------------
    cfg.MODEL.MASK_FORMER.DEPTH_ON = False
    cfg.MODEL.DEPTH_HEAD = CN()
    cfg.MODEL.DEPTH_HEAD.NUM_CONVS = 3
    cfg.MODEL.DEPTH_HEAD.CONVS_DIM = 256
    cfg.MODEL.DEPTH_HEAD.DEFORM = True
    cfg.MODEL.DEPTH_HEAD.COORD = True
    cfg.MODEL.DEPTH_HEAD.DEPTH_KERNEL_DIM = 16
    cfg.MODEL.DEPTH_HEAD.NORM = "GN"
    cfg.MODEL.DEPTH_HEAD.FACTOR = 5.0
    cfg.MODEL.DEPTH_HEAD.FACTOR_INS = 1.0
    cfg.INPUT.DEPTH_BOUND = True

    # ---------------------- 法线预测头部配置 ----------------------
    cfg.MODEL.MASK_FORMER.NORMAL_ON = False
    cfg.MODEL.Normal_HEAD = CN()
    cfg.MODEL.Normal_HEAD.NUM_CONVS = 3
    cfg.MODEL.Normal_HEAD.CONVS_DIM = 256
    cfg.MODEL.Normal_HEAD.DEFORM = True
    cfg.MODEL.Normal_HEAD.COORD = True
    cfg.MODEL.Normal_HEAD.Normal_KERNEL_DIM = 16
    cfg.MODEL.Normal_HEAD.NORM = "GN"
    cfg.MODEL.Normal_HEAD.FACTOR = 5.0
    cfg.MODEL.Normal_HEAD.FACTOR_INS = 1.0

    # ---------------------- 评估指标控制 ----------------------
    cfg.TEST.USE_REG_METRICS = False  # 是否使用回归指标进行评估（如 MAE、RMSE）

