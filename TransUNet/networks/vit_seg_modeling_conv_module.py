# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from . import pvt_v2 as pvt


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    从图像切块（patch）和位置嵌入（position embeddings）构建嵌入层
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2) # 将 3D 特征图展平为 2D
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        #print(embeddings.shape) #[24, 196, 768]
        embeddings = self.dropout(embeddings)
        #print(embeddings.shape)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis  # 是否可视化注意力权重
        self.layer = nn.ModuleList()  # 保存多个 Transformer 层
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)  # 最后归一化层
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            #print(layer_block) # 这里的layer_block返回的是Block？
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        #print(encoded, attn_weights)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        #print('Transfromer被执行 方为正常')
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # print('Transfromer被执行 方为正常')
        embedding_output, features = self.embeddings(input_ids) # 24, 196, 768
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        # print(f"必看DecoderBlock in_channels={in_channels}, out_channels={out_channels}, skip_channels={skip_channels}")
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 将skip的特征图与上采样特征图的空间尺度匹配
        self.skip_align = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        ) if skip_channels > 0 else None

        # self.cbam = CBAM(out_channels)

    def forward(self, x, skip=None):
        # print(f"\nup before x.shape={x.shape}")
        x = self.up(x)  # 上采样，将特征图分辨率扩大2倍
        # print(f"up after: x.shape={x.shape}")
        if skip is not None:  # 如果有跳跃连接的特征
            if self.skip_align is not None:
                skip = self.skip_align(skip)
                # print(f"after skip_align: skip.shape={skip.shape}")
            x = torch.cat([x, skip], dim=1)
            # print(f"必看After concatenation: x.shape={x.shape}, skip.shape={skip.shape}")
        x = self.conv1(x)  # 第一个卷积块处理
        x = self.conv2(x)  # 第二个卷积块处理
        # print(f"\nDecoderBlock 卷积处理结束: x.shape={x.shape}")
        # x = self.cbam(x)  # 添加CBAM
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 调整通道匹配PVT输出
        inner_channels = [512, 256, 128, 64]
        decoder_channels = [256, 128, 64, 16]
        skip_channels = [512, 256, 64, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(inner_channels, decoder_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features):
        # B, N, C = x.shape
        # h = w = int(N ** 0.5)
        # x = x.permute(0, 2, 1).view(B, C, h, w)  # (B, C, N) -> (B, C, h, w)

        for i, block in enumerate(self.blocks):
            x = block(x, skip=features[i] if i<len(features) else None)
        return x
       


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        # print(f"\n{x.shape}"); # [24, 1, 224, 224]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # print(f"\n{x.shape}"); # [24, 3, 224, 224]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print(f"self.decoder之前{x.shape}"); # [24, 196, 768]
        x = self.decoder(x, features)
        # print(f"self.decoder之后{x.shape}"); #[24, 16, 224, 224]

        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
               # print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

class PyramidAttentionTransfromerUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2):
        super().__init__()
        print("这里是conv_module")
        self.pvt = pvt.pvt_v2_b2(in_chans=3)
        
        # 特征适配器
        self.adapter = pvt.PVTAdapter(in_dim=512, target_patches=196)
        
        # 解码器调整(config似乎没用上)
        self.decoder = DecoderCup(config)
        # print(f"num_classes: {num_classes}")
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3
        )

        self.stage_convs = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1),  # 128->256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(320, 512, kernel_size=1),  # 256->512（原320->256改为320->512）
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Identity(),
        ])

        self.stage_attentions = nn.ModuleList([
            CBAM(64),   # s1维度
            CBAM(256),  # s2维度
            CBAM(512)   # s3维度
        ])

    def forward(self, x):
        # 处理单通道输入（灰度图转RGB）
        if x.size()[1] == 1:  # 检查通道维度是否为1
            x = x.repeat(1, 3, 1, 1)  # 复制单通道到三通道
            
        # PVT特征提取
        _ = self.pvt(x)  # 仅用于提取特征
        
        # 获取各阶段特征
        s1 = self.pvt.get_stage_features(1)  # [B,64,56,56]
        s2 = self.pvt.get_stage_features(2)  # [B,128,28,28]
        s3 = self.pvt.get_stage_features(3)  # [B,320,14,14]
        s4 = self.pvt.get_stage_features(4)  # [B,50,512]

        
        # 处理最后一层特征
        s4_adapted = self.adapter(s4)  # [B,196,512]

        # 通道对齐
        s1 = self.stage_attentions[0](self.stage_convs[0](s1))
        s2 = self.stage_attentions[1](self.stage_convs[1](s2))
        s3 = self.stage_attentions[2](self.stage_convs[2](s3))

        x = self.decoder(s4_adapted, [s3, s2, s1])
        logits = self.seg_head(x)
        return logits

    def load_from_pretrained(self, pretrained_path):
        """专门用于加载PVT预训练权重的方法"""
        print(f"正在从 {pretrained_path} 加载PVT预训练权重")
        if pretrained_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrained_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 检查预训练模型的键格式
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        
        # 提取原始pvt模型键
        pvt_state_dict = self.pvt.state_dict()
        print(f"原模型中的键数量: {len(pvt_state_dict.keys())}")
        print(f"\npvt_state_dict.keys()={pvt_state_dict.keys()}")
        new_state_dict = {}

        # 记录权重加载前后的状态变化
        key_params_before = {}
        for key in list(pvt_state_dict.keys())[:5]:
            if 'num_batches_tracked' not in key:
                key_params_before[key] = pvt_state_dict[key].clone().cpu().numpy().mean()

        # 处理可能的前缀差异
        prefix_to_try = ['', 'pvt.', 'backbone.', 'encoder.']
        matched_prefix = ''
        max_matched = 0
        
        # 找到最佳匹配前缀
        for prefix in prefix_to_try:
            matched = 0
            for k in checkpoint_model.keys():
                if k.startswith('head.'):
                    continue
                    
                # 尝试直接键匹配
                if f"{prefix}{k}" in pvt_state_dict:
                    matched += 1
                    
            if matched > max_matched:
                max_matched = matched
                matched_prefix = prefix
                
        print(f"最佳匹配前缀: '{matched_prefix}', 匹配键数量: {max_matched}")
        
        # 根据最佳前缀进行映射
        for k, v in checkpoint_model.items():
            # 跳过分类头
            if k.startswith('head.'):
                continue
                
            # 使用最佳前缀映射
            target_key = f"{matched_prefix}{k}"
            
            # PVT模型一般会在前面自动加pvt.，因此可能需要移除或添加
            if target_key in pvt_state_dict:
                new_state_dict[target_key] = v
            elif f"pvt.{target_key}" in pvt_state_dict:
                new_state_dict[f"pvt.{target_key}"] = v
            elif target_key.startswith('pvt.') and target_key[4:] in pvt_state_dict:
                new_state_dict[target_key[4:]] = v
                
        # 加载状态字典
        print(f"匹配的键数量: {len(new_state_dict)}")
        if new_state_dict:
            missing, unexpected = self.pvt.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载PVT预训练模型! 缺失键: {len(missing)}, 意外键: {len(unexpected)}")
            print(f"缺失键示例: {missing[:5] if missing else '无'}")
            
            # 检查权重变化
            for key in key_params_before:
                if key in new_state_dict:
                    current_mean = pvt_state_dict[key].cpu().numpy().mean()
                    before_mean = key_params_before[key]
                    print(f"参数 {key}: 变化前={before_mean:.6f}, 变化后={current_mean:.6f}")
                    if abs(current_mean - before_mean) > 1e-6:
                        print(f"  ✓ 参数已更新")
                    else:
                        print(f"  ✗ 参数未变化")
        else:
            print("警告: 未找到匹配的参数, 使用随机初始化")
            
        return new_state_dict



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid(self.conv(spatial))
        
        return x * channel * spatial

