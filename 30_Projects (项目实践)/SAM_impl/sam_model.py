import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VisionTransformer(nn.Module):
    """简化的视觉Transformer，用于图像编码"""

    def __init__(self, img_size=256, patch_size=16, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 计算patch数量
        self.num_patches = (img_size // patch_size) ** 2

        # Patch嵌入
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 修复：改进位置嵌入初始化
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch嵌入
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # 添加位置嵌入
        x = x + self.pos_embed

        # 应用Transformer块
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 重新整形为空间格式
        h = w = int(self.num_patches ** 0.5)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class ImageEncoder(nn.Module):
    """图像编码器，将图像编码为高维嵌入"""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 简化的ViT
        self.vit = VisionTransformer(img_size=256, patch_size=16, embed_dim=512)

        # 修复：改进颈部网络，使用GroupNorm替代LayerNorm
        self.neck = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """处理图像并返回特征图"""
        vit_features = self.vit(image)
        image_embedding = self.neck(vit_features)
        return image_embedding


class PromptEncoder(nn.Module):
    """提示编码器，将各种类型的提示编码为嵌入"""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 不同提示类型的学习嵌入
        self.point_embeddings = nn.Embedding(2, embed_dim)  # 0: 背景, 1: 前景
        self.box_embeddings = nn.Embedding(2, embed_dim)  # 0: 左上, 1: 右下
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # 无掩码提示时使用

        # 修复：预定义位置编码投影层
        self.pos_proj = nn.Linear(128, embed_dim)

        # 修复：预定义密集掩码编码层
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, padding=1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.point_embeddings.weight, std=0.02)
        nn.init.normal_(self.box_embeddings.weight, std=0.02)
        nn.init.normal_(self.no_mask_embed.weight, std=0.02)

    def _get_positional_encoding(self, coords: torch.Tensor, num_pos_feats=128) -> torch.Tensor:
        """坐标的傅里叶特征位置编码"""
        # 将坐标归一化到[-1, 1]
        coords = coords.float()
        coords = 2.0 * coords - 1.0

        # 创建频率带
        freq_bands = torch.arange(num_pos_feats // 4, device=coords.device, dtype=torch.float32)
        freq_bands = torch.pow(10000, -freq_bands / (num_pos_feats // 4))

        # 应用到x和y坐标
        x_coords = coords[..., 0:1]  # (..., 1)
        y_coords = coords[..., 1:2]  # (..., 1)

        x_embed = x_coords * freq_bands
        y_embed = y_coords * freq_bands

        # 连接sin和cos
        pos_embed = torch.cat([
            torch.sin(x_embed), torch.cos(x_embed),
            torch.sin(y_embed), torch.cos(y_embed)
        ], dim=-1)

        # 修复：使用预定义的投影层
        return self.pos_proj(pos_embed)

    def _encode_points(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """编码点提示"""
        pos_enc = self._get_positional_encoding(points)
        type_enc = self.point_embeddings(labels)
        return pos_enc + type_enc

    def _encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """编码框提示为两个角点"""
        # boxes形状: (B, num_boxes, 4) 为 (x_min, y_min, x_max, y_max)
        B, num_boxes, _ = boxes.shape

        # 提取左上角和右下角点
        tl_points = boxes[:, :, :2]  # (B, num_boxes, 2)
        br_points = boxes[:, :, 2:]  # (B, num_boxes, 2)

        # 获取位置编码
        tl_pos_enc = self._get_positional_encoding(tl_points)  # (B, num_boxes, embed_dim)
        br_pos_enc = self._get_positional_encoding(br_points)  # (B, num_boxes, embed_dim)

        # 获取类型编码
        tl_type_enc = self.box_embeddings.weight[0].unsqueeze(0).unsqueeze(0).expand(B, num_boxes, -1)
        br_type_enc = self.box_embeddings.weight[1].unsqueeze(0).unsqueeze(0).expand(B, num_boxes, -1)

        # 组合位置和类型编码
        tl_embed = tl_pos_enc + tl_type_enc
        br_embed = br_pos_enc + br_type_enc

        # 堆叠左上角和右下角嵌入
        # 输出形状: (B, num_boxes * 2, embed_dim)
        box_embeddings = torch.stack([tl_embed, br_embed], dim=2).flatten(1, 2)

        return box_embeddings

    def _encode_dense_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """通过卷积编码密集掩码提示"""
        # 修复：使用预定义的卷积层
        return self.mask_conv(mask)

    def forward(self, prompts: Dict) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """生成提示嵌入"""
        sparse_embeddings = []
        batch_size = prompts.get("batch_size", 1)

        if "points" in prompts and "point_labels" in prompts:
            point_embed = self._encode_points(prompts["points"], prompts["point_labels"])
            sparse_embeddings.append(point_embed)

        if "boxes" in prompts:
            box_embed = self._encode_boxes(prompts["boxes"])
            sparse_embeddings.append(box_embed)

        if sparse_embeddings:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = None

        if "mask" in prompts:
            dense_embedding = self._encode_dense_mask(prompts["mask"])
        else:
            # 创建"无掩码"嵌入
            dense_embedding = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(batch_size, -1, 16, 16)

        return sparse_embeddings, dense_embedding


class MLP(nn.Module):
    """标准MLP块"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderLayer(nn.Module):
    """轻量级掩码解码器的单层"""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = MLP(embed_dim, embed_dim * 2, embed_dim, dropout=dropout)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, tokens, image_embedding):
        # 展平图像嵌入用于注意力计算
        B, C, H, W = image_embedding.shape
        image_flat = image_embedding.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # 步骤1: token自注意力
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.token_self_attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        # 步骤2: token到图像的交叉注意力
        tokens_norm = self.norm2(tokens)
        attn_out, _ = self.cross_attn_token_to_image(tokens_norm, image_flat, image_flat)
        tokens = tokens + attn_out

        # 步骤3: token上的MLP
        tokens = tokens + self.mlp(self.norm3(tokens))

        # 步骤4: 图像到token的交叉注意力
        image_norm = self.norm4(image_flat)
        attn_out, _ = self.cross_attn_image_to_token(image_norm, tokens, tokens)
        image_flat = image_flat + attn_out

        # 重新整形图像
        updated_image_embedding = image_flat.transpose(1, 2).reshape(B, C, H, W)

        return tokens, updated_image_embedding


class MaskDecoder(nn.Module):
    """轻量级解码器，从图像和提示嵌入预测掩码"""

    def __init__(self, embed_dim=256, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()

        # 学习的输出token：3个掩码，1个IoU
        self.output_tokens = nn.Embedding(4, embed_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # 修复：改进上采样层，使用GroupNorm
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # MLP将输出token映射到动态线性分类器
        self.output_hypernet_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, dropout=dropout)

        self.iou_prediction_head = MLP(embed_dim, 256, 3, dropout=dropout)  # 3个掩码

        # 初始化权重
        nn.init.normal_(self.output_tokens.weight, std=0.02)

    def forward(self, image_embedding, sparse_prompt_embeddings, dense_prompt_embeddings):
        B = image_embedding.shape[0]

        # 准备输出token
        output_tokens = self.output_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        # 将稀疏提示与输出token合并
        if sparse_prompt_embeddings is not None:
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        else:
            tokens = output_tokens

        # 将密集提示嵌入添加到图像嵌入
        image_embedding = image_embedding + dense_prompt_embeddings

        # 运行解码器层
        for layer in self.layers:
            tokens, image_embedding = layer(tokens, image_embedding)

        # 上采样图像嵌入
        upscaled_image_embedding = self.upsample_layers(image_embedding)  # (B, 32, 64, 64)

        # 超网络预测
        hypernet_in = tokens[:, :4]  # 前4个token（3个掩码 + 1个IoU）
        mask_weights = self.output_hypernet_mlp(hypernet_in[:, :3])  # (B, 3, 32)

        # 空间点积获得掩码
        mask_weights = mask_weights.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 32, 1, 1)
        upscaled_image_embedding = upscaled_image_embedding.unsqueeze(1)  # (B, 1, 32, 64, 64)

        masks = (mask_weights * upscaled_image_embedding).sum(dim=2)  # (B, 3, 64, 64)

        # 预测IoU分数
        iou_token_out = hypernet_in[:, 3]
        iou_predictions = self.iou_prediction_head(iou_token_out)  # (B, 3)

        return masks, iou_predictions


class SAM(nn.Module):
    """SAM模型主类"""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.prompt_encoder = PromptEncoder(embed_dim)
        self.mask_decoder = MaskDecoder(embed_dim)

        self._precomputed_embedding = None
        self._original_image_size = None

    def preprocess(self, image):
        """预处理图像为tensor格式"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # 添加batch维度

        if image.shape[1] != 3:  # 如果通道在最后
            image = image.permute(0, 3, 1, 2)

        # 归一化到[0, 1]
        if image.max() > 1:
            image = image / 255.0

        # 调整大小到256x256
        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)

        return image

    def set_image(self, image):
        """预计算图像嵌入"""
        if isinstance(image, np.ndarray):
            self._original_image_size = image.shape[:2]
        else:
            self._original_image_size = image.shape[-2:]

        processed_image = self.preprocess(image)

        with torch.no_grad():
            self._precomputed_embedding = self.image_encoder(processed_image)

    def predict(self, prompts: Dict) -> Dict:
        """给定提示，预测预设图像的掩码"""
        if self._precomputed_embedding is None:
            raise RuntimeError("必须在predict()之前调用set_image()")

        # 如果提示中没有batch_size，添加它
        prompts["batch_size"] = self._precomputed_embedding.shape[0]

        with torch.no_grad():
            # 编码提示
            sparse_embed, dense_embed = self.prompt_encoder(prompts)

            # 解码掩码
            low_res_masks, iou_scores = self.mask_decoder(
                self._precomputed_embedding,
                sparse_embed,
                dense_embed
            )

            # 后处理掩码（上采样到原始尺寸）
            final_masks = F.interpolate(
                low_res_masks,
                size=self._original_image_size,
                mode='bilinear',
                align_corners=False
            )

        return {"masks": final_masks, "iou_scores": iou_scores}

    def forward(self, image, prompts):
        """训练时的前向传播"""
        # 编码图像
        image_embedding = self.image_encoder(image)

        # 添加batch_size到提示
        prompts["batch_size"] = image.shape[0]

        # 编码提示
        sparse_embed, dense_embed = self.prompt_encoder(prompts)

        # 解码掩码
        masks, iou_scores = self.mask_decoder(image_embedding, sparse_embed, dense_embed)

        return masks, iou_scores


# 修复：改进损失函数
def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """用于处理类别不平衡的焦点损失"""
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def dice_loss(pred, target, smooth=1e-5):
    """分割任务的Dice损失"""
    pred = torch.sigmoid(pred)

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

    return 1 - dice.mean()


def calculate_iou(pred, target):
    """计算预测和目标之间的IoU"""
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))

    iou = intersection / (union + 1e-8)
    return iou


# 修复：添加自适应损失权重
class AdaptiveLossWeights(nn.Module):
    """自适应损失权重"""

    def __init__(self):
        super().__init__()
        self.focal_weight = nn.Parameter(torch.tensor(1.0))
        self.dice_weight = nn.Parameter(torch.tensor(1.0))
        self.iou_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, focal_loss, dice_loss, iou_loss):
        total_loss = (torch.exp(self.focal_weight) * focal_loss +
                      torch.exp(self.dice_weight) * dice_loss +
                      torch.exp(self.iou_weight) * iou_loss)
        return total_loss