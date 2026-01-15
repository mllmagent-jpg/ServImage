"""
模型变体4的架构修改实现 - CBM (Concept Bottleneck Model) - 多任务分类版
直接预测三维度概念分数，不进行概念融合和最终分类

统一标签格式：7个独立的6分类任务
- 7个质量指标，每个都是独立的6分类问题
- 6个类别：1分、2分、3分、4分、5分、N/A
- 标签值: 1, 2, 3, 4, 5 (离散分数) 或 -1 (N/A)
- 模型输出: 每个指标输出6维logits，对应6个类别

数据集特征（7个独立的6分类任务）：
- BRF (Brief Requirement Fulfillment): 需求完成度
  - brf: 1-5分（6分类：类别0-4对应1-5分，类别5对应N/A）

- VEQ (Visual Evaluation Quality): 视觉质量评估（4个指标）
  - veq_clarity: 1-5分或N/A (清晰度)
  - veq_realism: 1-5分或N/A (真实感)
  - veq_aesthetic: 1-5分或N/A (美学)
  - veq_text: 1-5分或N/A (文字质量)

- CNS (Consistency): 一致性（2个指标）
  - cns_edit: 1-5分或N/A (编辑一致性)
  - cns_set: 1-5分或N/A (集合一致性)

注意：
- 如果原始数据是连续分数，需要先四舍五入到1-5的整数
- N/A标记为-1，在loss计算时会被过滤
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from peft import LoraConfig, get_peft_model, TaskType


class AttentionPooling(nn.Module):
    """注意力池化层 - 使用可学习的查询向量对序列进行加权平均"""

    def __init__(self, hidden_size):
        super().__init__()
        # 可学习的查询向量
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        # 缩放因子
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            pooled: (batch_size, hidden_size) - 注意力加权后的向量
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 扩展查询向量到batch_size
        query = self.attention_query.expand(batch_size, -1, -1)  # (B, 1, H)

        # 计算注意力分数: query @ keys^T
        attn_scores = torch.bmm(query, hidden_states.transpose(1, 2))  # (B, 1, S)
        attn_scores = attn_scores * self.scale

        # Softmax归一化得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, S)

        # 使用注意力权重对序列进行加权求和
        pooled = torch.bmm(attn_weights, hidden_states)  # (B, 1, H)
        pooled = pooled.squeeze(1)  # (B, H)

        return pooled


class CBMQualityDecoder(nn.Module):
    """
    概念瓶颈模型 (Concept Bottleneck Model) 质量解码器 - 多任务分类版

    架构设计：
    1. 注意力池化层：使用注意力机制对整个序列进行池化
    2. 共享特征提取器：从 VLM hidden states 提取通用特征
    3. 三个独立的概念预测分支：
       - BRF 分支：预测需求完成度（1个指标）
       - VEQ 分支：预测视觉质量的各个维度（4个指标）
       - CNS 分支：预测一致性的各个维度（2个指标）

    输出：7个独立的6分类预测头
    - 每个预测头输出 6 维 logits
    - 6个类别对应：1分、2分、3分、4分、5分、N/A
    - 类别索引映射：索引0-4对应1-5分，索引5对应N/A（目前不使用）
    """

    def __init__(
        self,
        hidden_size=3584,           # Qwen3-VL-2B hidden size
        concept_hidden_size=512,    # 概念层维度
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.concept_hidden_size = concept_hidden_size

        # 统一的预测类别数: 6 (对应 1-5分 + N/A)
        # 注意：实际使用中主要用前5个类别（0-4对应1-5分），第6个类别（索引5）保留给N/A
        self.num_classes = 6

        # ==================== 0. 注意力池化层 ====================
        self.attention_pooling = AttentionPooling(hidden_size)

        # ==================== 1. 共享特征提取器 ====================
        # 从 hidden_size 压缩到中间表示
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ==================== 2. BRF 概念预测分支（1个指标） ====================
        self.brf_concept_encoder = nn.Sequential(
            nn.Linear(1024, concept_hidden_size),
            nn.LayerNorm(concept_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # BRF 预测头: 6分类 (1-5分 + N/A)
        self.brf_head = nn.Linear(concept_hidden_size, self.num_classes)

        # ==================== 3. VEQ 概念预测分支（4个指标） ====================
        self.veq_concept_encoder = nn.Sequential(
            nn.Linear(1024, concept_hidden_size),
            nn.LayerNorm(concept_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # VEQ 各维度预测头: 每个都是 6分类 (1-5分 + N/A)
        self.veq_clarity_head = nn.Linear(concept_hidden_size, self.num_classes)
        self.veq_realism_head = nn.Linear(concept_hidden_size, self.num_classes)
        self.veq_aesthetic_head = nn.Linear(concept_hidden_size, self.num_classes)
        self.veq_text_head = nn.Linear(concept_hidden_size, self.num_classes)

        # ==================== 4. CNS 概念预测分支（2个指标） ====================
        self.cns_concept_encoder = nn.Sequential(
            nn.Linear(1024, concept_hidden_size),
            nn.LayerNorm(concept_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CNS 两个维度预测头: 每个都是 6分类 (1-5分 + N/A)
        self.cns_edit_head = nn.Linear(concept_hidden_size, self.num_classes)
        self.cns_set_head = nn.Linear(concept_hidden_size, self.num_classes)

    def forward(self, hidden_states):
        """
        前向传播

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            concepts: 7个质量指标的预测，每个都是6分类的logits
            {
                'brf': (B, 6),           # BRF指标的6分类logits
                'veq_clarity': (B, 6),   # VEQ清晰度的6分类logits
                'veq_realism': (B, 6),   # VEQ真实感的6分类logits
                'veq_aesthetic': (B, 6), # VEQ美学的6分类logits
                'veq_text': (B, 6),      # VEQ文字质量的6分类logits
                'cns_edit': (B, 6),      # CNS编辑一致性的6分类logits
                'cns_set': (B, 6),       # CNS集合一致性的6分类logits
            }
        """
        # 使用注意力池化对整个序列进行加权平均
        pooled_hidden = self.attention_pooling(hidden_states)  # (batch_size, hidden_size)

        # 1. 共享特征提取
        shared_features = self.shared_encoder(pooled_hidden)  # (B, 1024)

        # ==================== 2. BRF 概念预测（1个6分类任务） ====================
        brf_features = self.brf_concept_encoder(shared_features)  # (B, 512)
        brf_logits = self.brf_head(brf_features)  # (B, 6)

        # ==================== 3. VEQ 概念预测（4个6分类任务） ====================
        veq_features = self.veq_concept_encoder(shared_features)  # (B, 512)
        veq_clarity_logits = self.veq_clarity_head(veq_features)  # (B, 6)
        veq_realism_logits = self.veq_realism_head(veq_features)  # (B, 6)
        veq_aesthetic_logits = self.veq_aesthetic_head(veq_features)  # (B, 6)
        veq_text_logits = self.veq_text_head(veq_features)  # (B, 6)

        # ==================== 4. CNS 概念预测（2个6分类任务） ====================
        cns_features = self.cns_concept_encoder(shared_features)  # (B, 512)
        cns_edit_logits = self.cns_edit_head(cns_features)  # (B, 6)
        cns_set_logits = self.cns_set_head(cns_features)  # (B, 6)

        # ==================== 5. 返回7个独立的6分类预测 ====================
        return {
            'brf': brf_logits,  # (B, 6): 1-5分 + N/A
            'veq_clarity': veq_clarity_logits,  # (B, 6): 1-5分 + N/A
            'veq_realism': veq_realism_logits,  # (B, 6): 1-5分 + N/A
            'veq_aesthetic': veq_aesthetic_logits,  # (B, 6): 1-5分 + N/A
            'veq_text': veq_text_logits,  # (B, 6): 1-5分 + N/A
            'cns_edit': cns_edit_logits,  # (B, 6): 1-5分 + N/A
            'cns_set': cns_set_logits,  # (B, 6): 1-5分 + N/A
        }


def compute_cbm_loss(concepts, concept_labels):
    """
    计算 CBM 的概念预测损失 - 7个独立的6分类任务

    任务设定：
    - 7个质量指标：brf, veq_clarity, veq_realism, veq_aesthetic, veq_text, cns_edit, cns_set
    - 每个指标是独立的6分类问题：1分、2分、3分、4分、5分、N/A
    - 对每个有效指标独立计算交叉熵损失，然后平均

    标签格式：
    concept_labels = {
        'brf': Tensor (B,),          # 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'veq_clarity': Tensor (B,),  # 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'veq_realism': Tensor (B,),  # 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'veq_aesthetic': Tensor (B,),# 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'veq_text': Tensor (B,),     # 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'cns_edit': Tensor (B,),     # 值为 1, 2, 3, 4, 5 或 -1(N/A)
        'cns_set': Tensor (B,),      # 值为 1, 2, 3, 4, 5 或 -1(N/A)
    }

    标签映射规则：
    - 标签值 1 → 类别索引 0
    - 标签值 2 → 类别索引 1
    - 标签值 3 → 类别索引 2
    - 标签值 4 → 类别索引 3
    - 标签值 5 → 类别索引 4
    - 标签值 -1(N/A) → 过滤，不参与loss计算

    Args:
        concepts: 模型输出的概念字典 {维度名: logits (B, 6)}
        concept_labels: 概念标签字典 {维度名: labels (B,)}

    Returns:
        total_loss: 总损失（所有有效指标的平均交叉熵）
        loss_dict: 各项损失的字典（用于日志）
    """
    import torch

    if not isinstance(concept_labels, dict):
        raise ValueError(
            f"concept_labels 必须是字典格式，收到类型: {type(concept_labels)}\n"
            f"期望格式: {{'brf': Tensor, 'veq_clarity': Tensor, ...}}\n"
            f"7个指标的值应为 1-5 (离散分数) 或 -1 (N/A)"
        )

    loss_dict = {}
    total_loss = 0.0
    num_losses = 0

    # 定义所有概念维度（7个独立的6分类任务）
    all_dims = ['brf', 'veq_clarity', 'veq_realism', 'veq_aesthetic', 'veq_text', 'cns_edit', 'cns_set']

    # ==================== 统一处理7个独立的6分类任务 ====================
    for dim_name in all_dims:
        # 检查该维度是否存在标签
        if dim_name not in concept_labels:
            continue

        # 检查该维度是否存在预测
        if dim_name not in concepts:
            print(f"[Warning] 标签中存在 '{dim_name}'，但模型输出中不存在")
            continue

        labels = concept_labels[dim_name]  # (B,)
        logits = concepts[dim_name]  # (B, 6)

        # 过滤有效样本 (标签值 >= 1，排除 N/A=-1 和其他无效值)
        valid_mask = labels >= 1

        if valid_mask.sum() > 0:
            # 提取有效样本
            valid_labels = labels[valid_mask].long()  # (N,)
            valid_logits = logits[valid_mask]  # (N, 6)

            # 标签映射: 1-5 → 0-4
            # clamp 确保标签在合法范围内 [1, 5]
            valid_labels = (valid_labels - 1).clamp(0, 4)

            # 计算交叉熵损失
            loss = F.cross_entropy(valid_logits, valid_labels)

            total_loss += loss
            loss_dict[f'{dim_name}_loss'] = loss.item()
            num_losses += 1

    # ==================== 返回平均损失 ====================
    if num_losses > 0:
        total_loss = total_loss / num_losses
        loss_dict['num_losses'] = num_losses
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    else:
        # 没有任何有效损失
        device = next(iter(concepts.values())).device
        loss_dict['total_loss'] = 0.0
        loss_dict['num_losses'] = 0
        return torch.tensor(0.0, device=device), loss_dict


def variant4_forward(
    model,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    concept_labels=None,  # 概念标签
    scores=None,  # 评分标签（与concept_labels同义）
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    use_cache=None,
    **kwargs
):
    """
    Variant4 CBM 的自定义 forward（简化版）

    整合了：
    1. Qwen3VL 的原始forward（获取hidden states）
    2. CBMQualityDecoder 的概念预测
    3. 概念损失计算
    """

    # 递归检测
    if hasattr(model, '_in_variant4_forward') and model._in_variant4_forward:
        raise RecursionError("Detected recursive call to variant4_forward!")

    model._in_variant4_forward = True

    try:
        # 处理scores参数：如果提供了scores但没有concept_labels，使用scores作为concept_labels
        if scores is not None and concept_labels is None:
            concept_labels = scores

        # 重要：移除 kwargs 中的 labels，避免传递给原始 forward
        # 因为我们使用 concept_labels 而不是语言模型的 token labels
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'labels'}

        # 1. 调用原始forward获取输出（不传入labels，只获取hidden states）
        outputs = model._original_forward_func(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
            labels=None,  # 显式设置为 None，不计算语言模型loss
            **kwargs_filtered
        )

        # 2. 提取最后一层hidden states
        hidden_states = outputs.hidden_states[-1]

        # 3. CBM Decoder 预测所有概念
        concepts = model.quality_decoder(hidden_states)

        # 4. 计算Loss
        loss = None
        if concept_labels is not None:
            loss, loss_dict = compute_cbm_loss(concepts, concept_labels)
            model._last_loss_dict = loss_dict

        # 5. 返回输出（logits 返回概念预测）
        from transformers.modeling_outputs import CausalLMOutputWithPast
        # 关键优化：不返回hidden_states和attentions以节省GPU内存
        # 在评估时，这些中间结果会累积导致OOM
        return CausalLMOutputWithPast(
            loss=loss,
            logits=concepts,  # 返回概念预测字典
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=None,  # 不返回，避免GPU内存累积
            attentions=None,  # 不返回，避免GPU内存累积
        )
    finally:
        model._in_variant4_forward = False


def apply_variant4_modifications(model, config=None):
    """
    应用 Variant4 CBM 的架构修改（简化版）
    """
    if config is None:
        config = {}

    if hasattr(model, '_variant4_modified') and model._variant4_modified:
        print("[Variant4] 模型已经应用过修改，跳过")
        return model

    if hasattr(model.forward, '__name__') and model.forward.__name__ == 'custom_forward':
        print("[Variant4] Forward 已被包装，跳过")
        return model

    hidden_size = model.config.text_config.hidden_size
    concept_hidden_size = config.get('concept_hidden_size', 512)
    dropout = config.get('decoder_dropout', 0.1)

    # 添加 CBM Decoder
    model.quality_decoder = CBMQualityDecoder(
        hidden_size=hidden_size,
        concept_hidden_size=concept_hidden_size,
        dropout=dropout,
    )

    if hasattr(model, '_init_weights'):
        model.quality_decoder.apply(model._init_weights)

    if not hasattr(model, '_original_forward_func'):
        from transformers import Qwen3VLForConditionalGeneration
        model._original_forward_func = Qwen3VLForConditionalGeneration.forward

    model._variant4_config = config

    def custom_forward(*args, **kwargs):
        return variant4_forward(model, *args, **kwargs)

    model.forward = custom_forward
    model._variant4_modified = True

    print(f"[Variant4] CBMQualityDecoder (多任务分类版) 已添加:")
    print(f"  - 任务设定: 7个独立的6分类任务")
    print(f"  - 指标: BRF(1个) + VEQ(4个) + CNS(2个) = 7个")
    print(f"  - 每个指标: 6个类别 (1分、2分、3分、4分、5分、N/A)")
    print(f"  - 模型输出: 7个预测头 × 6维logits")
    print(f"  - 标签格式: {{维度名: 1-5 或 -1(N/A)}}")
    print(f"  - 损失函数: 对每个指标独立计算交叉熵，然后平均")

    return model


def apply_lora_to_variant4(model, config=None):
    """为 Variant4 应用 LoRA"""
    if config is None:
        from .config import get_variant4_config
        config = get_variant4_config()

    vision_lora_enabled = config.get('lora_vision_enable', True)
    projector_lora_enabled = config.get('lora_projector_enable', True)
    llm_lora_enabled = config.get('lora_llm_enable', False)

    if not vision_lora_enabled and not projector_lora_enabled and not llm_lora_enabled:
        print("[Variant4 LoRA] 已禁用，仅训练 CBM Decoder")
        for name, param in model.named_parameters():
            param.requires_grad = 'quality_decoder' in name
        return model

    for param in model.parameters():
        param.requires_grad = False

    # ==================== Vision Encoder LoRA ====================
    if vision_lora_enabled:
        vision_lora_config = LoraConfig(
            r=config.get('lora_vision_r', 8),
            lora_alpha=config.get('lora_vision_alpha', 16),
            lora_dropout=config.get('lora_vision_dropout', 0.05),
            target_modules=["qkv", "proj"],
            bias="none",
        )
        model.visual = get_peft_model(model.visual, vision_lora_config)
        print(f"[Variant4 LoRA] Vision Encoder - r={config['lora_vision_r']}, alpha={config['lora_vision_alpha']}")

    # ==================== Projector (visual.merger) LoRA ====================
    if projector_lora_enabled:
        if hasattr(model.visual, 'merger'):
            # Qwen3VL的merger结构: norm + linear_fc1 + act_fn + linear_fc2
            # 我们对两个Linear层应用LoRA
            projector_lora_config = LoraConfig(
                r=config.get('lora_projector_r', 8),
                lora_alpha=config.get('lora_projector_alpha', 16),
                lora_dropout=config.get('lora_projector_dropout', 0.05),
                target_modules=["linear_fc1", "linear_fc2"],  # Qwen3VL的两个Linear层
                bias="none",
            )
            # 不使用try-except，让错误直接抛出以确保所有ranks行为一致
            # 如果LoRA应用失败，训练应该停止而不是默默降级到全参数训练
            model.visual.merger = get_peft_model(model.visual.merger, projector_lora_config)
            print(f"[Variant4 LoRA] Projector LoRA 已应用 - r={config['lora_projector_r']}, alpha={config['lora_projector_alpha']}")
        else:
            print("[Variant4 LoRA] 未找到 visual.merger，跳过 Projector LoRA")
    else:
        # 如果禁用 Projector LoRA，冻结它
        if hasattr(model.visual, 'merger'):
            for param in model.visual.merger.parameters():
                param.requires_grad = False
            print("[Variant4 LoRA] Projector - 保持冻结状态")

    # ==================== Language Model LoRA ====================
    if llm_lora_enabled:
        if not hasattr(model.language_model, "prepare_inputs_for_generation"):
            def _dummy(*args, **kwargs):
                raise NotImplementedError("Generation not supported")
            model.language_model.prepare_inputs_for_generation = _dummy

        llm_lora_config = LoraConfig(
            r=config.get('lora_llm_r', 8),
            lora_alpha=config.get('lora_llm_alpha', 16),
            lora_dropout=config.get('lora_llm_dropout', 0.05),
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.language_model = get_peft_model(model.language_model, llm_lora_config)
        print(f"[Variant4 LoRA] LLM - r={config['lora_llm_r']}, alpha={config['lora_llm_alpha']}")
    else:
        for param in model.language_model.parameters():
            param.requires_grad = False
        print("[Variant4 LoRA] LLM - 保持冻结状态")

    # ==================== CBM Decoder ====================
    if hasattr(model, 'quality_decoder'):
        for param in model.quality_decoder.parameters():
            param.requires_grad = True
        print("[Variant4 LoRA] CBM Decoder - 全参数训练")

    return model


def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_params = 0

    print("\n[Variant4] 可训练参数:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora' in name.lower() or 'quality_decoder' in name:
                print(f"  ✓ {name}: {param.numel():,}")

    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"\n[Variant4] 总计: {trainable_params:,} / {all_params:,} ({pct:.2f}%)")

    return trainable_params, all_params
