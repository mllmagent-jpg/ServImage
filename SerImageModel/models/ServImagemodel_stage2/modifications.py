"""
Stage2 Modifications - 基于Stage1权重，使用base模型分类头
- 加载Stage1的所有权重（base model + LoRA）
- 移除Stage1的quality_decoder（7维质量评估头）
- 使用base模型的SingleTaskQualityDecoder（二分类头）
- 添加LoRA继续训练
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

        # 扩展查询向量到batch_size，并确保数据类型匹配
        query = self.attention_query.expand(batch_size, -1, -1).to(hidden_states.dtype)  # (B, 1, H)

        # 计算注意力分数: query @ keys^T
        attn_scores = torch.bmm(query, hidden_states.transpose(1, 2))  # (B, 1, S)
        attn_scores = attn_scores * self.scale

        # Softmax归一化得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, S)

        # 使用注意力权重对序列进行加权求和
        pooled = torch.bmm(attn_weights, hidden_states)  # (B, 1, H)
        pooled = pooled.squeeze(1)  # (B, H)

        return pooled


class SingleTaskQualityDecoder(nn.Module):
    """
    单任务质量评估解码器（从base模型复制）
    二分类：Accept (1) / Reject (0)

    架构（5层MLP金字塔结构）：
    - 输入：使用注意力池化对整个序列进行加权平均
    - 5层MLP：hidden_size -> 1024 -> 256 -> 64 -> 16 -> 2
    - 每层使用ReLU激活函数
    - 逐步4倍降维，保持平滑的特征转换
    """
    def __init__(self, hidden_size=3584, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 注意力池化层
        self.attention_pooling = AttentionPooling(hidden_size)

        # 5层MLP分类头
        self.fc1 = nn.Linear(hidden_size, 1024)   # 层1: hidden_size -> 1024
        self.fc2 = nn.Linear(1024, 256)            # 层2: 1024 -> 256
        self.fc3 = nn.Linear(256, 64)              # 层3: 256 -> 64
        self.fc4 = nn.Linear(64, 16)               # 层4: 64 -> 16
        self.fc5 = nn.Linear(16, 2)                # 层5: 16 -> 2 (二分类输出)

        self.relu = nn.ReLU()

        # 添加logits缩放因子，避免输出过小
        self.logits_scale = 100.0  # 将logits放大100倍

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化MLP权重
        使用Kaiming初始化，适合ReLU激活函数
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states):
        """
        前向传播

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            logits: (batch_size, 2) - 二分类输出
        """
        # 使用注意力池化对整个序列进行加权平均
        pooled_hidden = self.attention_pooling(hidden_states)  # (batch_size, hidden_size)

        # 5层MLP逐步降维
        x = self.relu(self.fc1(pooled_hidden))  # (batch_size, 1024)
        x = self.relu(self.fc2(x))            # (batch_size, 256)
        x = self.relu(self.fc3(x))            # (batch_size, 64)
        x = self.relu(self.fc4(x))            # (batch_size, 16)
        logits = self.fc5(x)                  # (batch_size, 2)

        # 应用缩放因子，增强信号强度
        logits = logits * self.logits_scale

        return logits


def compute_quality_loss(logits, labels):
    """
    计算质量评估损失（二分类）

    Args:
        logits: (batch_size, 2)
        labels: (batch_size,) - 值为0/1

    Returns:
        loss: 标量
    """
    import os
    device = logits.device

    # 从环境变量读取权重配置 (默认1:1平衡权重)
    reject_weight = float(os.getenv('CLASS_WEIGHT_REJECT', '1.0'))
    accept_weight = float(os.getenv('CLASS_WEIGHT_ACCEPT', '1.0'))

    class_weights = torch.tensor([reject_weight, accept_weight], dtype=logits.dtype, device=device)

    loss = F.cross_entropy(logits, labels, weight=class_weights)
    return loss


def stage2_forward(
    model,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    concept_labels=None,  # collator使用concept_labels而不是labels
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
    Stage2 的自定义 forward

    整合了：
    1. Qwen3VL 的原始forward（获取hidden states）
    2. SingleTaskQualityDecoder 的二分类预测
    3. 简化的 loss 计算
    """

    # 递归检测
    if hasattr(model, '_in_stage2_forward') and model._in_stage2_forward:
        raise RecursionError("Detected recursive call to stage2_forward!")

    model._in_stage2_forward = True

    try:
        # 使用concept_labels（如果提供了）作为labels
        quality_labels = concept_labels if concept_labels is not None else labels

        # 1. 调用原始forward获取输出
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
            output_hidden_states=True,  # 必须获取hidden states
            return_dict=True,
            use_cache=use_cache,
            **kwargs
        )

        # 2. 提取最后一层hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # 3. Quality Decoder 预测
        quality_logits = model.quality_decoder(hidden_states)  # (batch_size, 2)

        # 4. 计算Loss（如果提供了labels）
        loss = None
        if quality_labels is not None:
            loss = compute_quality_loss(quality_logits, quality_labels)

        # 5. 返回自定义输出
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=quality_logits,  # 返回质量预测结果
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=None,  # 不返回，避免GPU内存累积
            attentions=None,  # 不返回，避免GPU内存累积
        )
    finally:
        model._in_stage2_forward = False


def apply_stage2_modifications(model, config=None):
    """
    应用 Stage2 的架构修改

    修改内容：
    1. 添加 SingleTaskQualityDecoder（从base模型）
    2. 重写 forward 方法以支持质量评估
    3. 添加自定义 loss 计算

    Args:
        model: Qwen3VLForConditionalGeneration 实例（已加载Stage1权重）
        config: 配置字典
    """
    if config is None:
        config = {}

    # 检查是否已经应用过修改
    if hasattr(model, '_stage2_modified') and model._stage2_modified:
        print("[Stage2] 模型已经应用过 Stage2 修改，跳过")
        return model

    if hasattr(model.forward, '__name__') and model.forward.__name__ == 'custom_forward':
        print("[Stage2] Forward 方法已被包装，跳过")
        return model

    hidden_size = model.config.text_config.hidden_size
    dropout = config.get('decoder_dropout', 0.1)

    # 1. 移除Stage1的quality_decoder（如果存在）
    if hasattr(model, 'quality_decoder'):
        print("[Stage2] 移除 Stage1 的 quality_decoder...")
        delattr(model, 'quality_decoder')

    # 2. 添加 SingleTaskQualityDecoder（从base模型）
    model.quality_decoder = SingleTaskQualityDecoder(
        hidden_size=hidden_size,
        dropout=dropout
    )
    print(f"[Stage2] 添加新的 SingleTaskQualityDecoder (二分类头)")

    # 3. 初始化新参数
    if hasattr(model, '_init_weights'):
        model.quality_decoder.apply(model._init_weights)

    # 4. 保存原始forward方法
    if not hasattr(model, '_original_forward_func'):
        from transformers import Qwen3VLForConditionalGeneration
        model._original_forward_func = Qwen3VLForConditionalGeneration.forward
        print("[Stage2] 已保存原始 forward 方法")

    # 5. 替换forward方法
    def custom_forward(*args, **kwargs):
        return stage2_forward(model, *args, **kwargs)

    model.forward = custom_forward

    # 6. 标记已修改
    model._stage2_modified = True

    # 移除Stage1的标记（如果存在）
    if hasattr(model, '_variant4_modified'):
        delattr(model, '_variant4_modified')

    print(f"[Stage2] 已完成架构修改:")
    print(f"  - 移除: Stage1的CBMQualityDecoder（7维质量评估）")
    print(f"  - 添加: SingleTaskQualityDecoder（二分类: Accept/Reject）")
    print(f"  - Hidden size: {hidden_size}, Dropout: {dropout}")

    return model


def apply_lora_to_stage2(model, config=None):
    """
    为 Stage2 应用新的 LoRA（在Stage1 LoRA基础上再添加LoRA）

    策略：
    1. **冻结Stage1的所有权重（包括Stage1的LoRA）** - 这是关键！
    2. 在Stage1 LoRA基础上再添加新的Stage2 LoRA
    3. 只训练Stage2的新LoRA和新的Quality Decoder
    4. Stage1的权重完全冻结，不参与训练

    LoRA 应用位置：
    1. Visual Encoder (model.visual) - 添加新LoRA（Stage1 LoRA冻结）
    2. Projector (model.visual.merger) - 添加新LoRA（Stage1 LoRA冻结）
    3. Language Model (model.language_model) - 添加新LoRA（Stage1 LoRA冻结）
    4. Quality Decoder - 新初始化，全参数训练
    """
    if config is None:
        from .config import get_stage2_config
        config = get_stage2_config()

    vision_lora_enabled = config.get('lora_vision_enable', True)
    projector_lora_enabled = config.get('lora_projector_enable', True)
    llm_lora_enabled = config.get('lora_llm_enable', True)

    print("\n[Stage2 LoRA] ========================================")
    print("[Stage2 LoRA] 配置LoRA训练策略")
    print("[Stage2 LoRA] ========================================")
    print(f"  - 策略：冻结Stage1所有权重，在其基础上添加新LoRA")
    print(f"  - Vision LoRA: {'启用（添加新LoRA）' if vision_lora_enabled else '禁用'}")
    print(f"  - Projector LoRA: {'启用（添加新LoRA）' if projector_lora_enabled else '禁用'}")
    print(f"  - LLM LoRA: {'启用（添加新LoRA）' if llm_lora_enabled else '禁用'}")

    # ==================== 步骤1: 冻结所有参数（包括Stage1的LoRA） ====================
    print("\n[Stage2 LoRA] 步骤1: 冻结Stage1所有权重...")
    frozen_count = 0
    stage1_lora_count = 0

    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen_count += param.numel()
        if 'lora' in name.lower():
            stage1_lora_count += param.numel()

    print(f"[Stage2 LoRA] ✓ 已冻结所有参数: {frozen_count:,}")
    print(f"[Stage2 LoRA] ✓ 其中Stage1 LoRA参数: {stage1_lora_count:,}")

    # ==================== 步骤2: 在冻结的Stage1基础上添加新LoRA ====================
    print("\n[Stage2 LoRA] 步骤2: 在Stage1 LoRA基础上添加新LoRA...")

    # ==================== Vision Encoder LoRA ====================
    if vision_lora_enabled and hasattr(model, 'visual'):
        vision_lora_config = LoraConfig(
            r=config.get('lora_vision_r', 8),
            lora_alpha=config.get('lora_vision_alpha', 16),
            lora_dropout=config.get('lora_vision_dropout', 0.05),
            target_modules=["qkv", "proj"],
            bias="none",
        )

        # 在已有的Stage1 LoRA基础上再添加新的LoRA
        try:
            model.visual = get_peft_model(model.visual, vision_lora_config)
            print(f"[Stage2 LoRA] ✓ Vision Encoder - 已添加新LoRA（r={config['lora_vision_r']}, alpha={config['lora_vision_alpha']}）")
        except Exception as e:
            print(f"[Stage2 LoRA] ⚠ Vision Encoder - 添加LoRA失败: {e}")
            print(f"[Stage2 LoRA] ⚠ 保持Stage1 LoRA冻结状态")

    # ==================== Projector (visual.merger) LoRA ====================
    if projector_lora_enabled and hasattr(model.visual, 'merger'):
        projector_lora_config = LoraConfig(
            r=config.get('lora_projector_r', 8),
            lora_alpha=config.get('lora_projector_alpha', 16),
            lora_dropout=config.get('lora_projector_dropout', 0.05),
            target_modules=["linear_fc1", "linear_fc2"],
            bias="none",
        )

        try:
            model.visual.merger = get_peft_model(model.visual.merger, projector_lora_config)
            print(f"[Stage2 LoRA] ✓ Projector - 已添加新LoRA（r={config['lora_projector_r']}, alpha={config['lora_projector_alpha']}）")
        except Exception as e:
            print(f"[Stage2 LoRA] ⚠ Projector - 添加LoRA失败: {e}")
            print(f"[Stage2 LoRA] ⚠ 保持Stage1 LoRA冻结状态")

    # ==================== Language Model LoRA ====================
    if llm_lora_enabled and hasattr(model, 'language_model'):
        # 确保有prepare_inputs_for_generation方法
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

        try:
            model.language_model = get_peft_model(model.language_model, llm_lora_config)
            print(f"[Stage2 LoRA] ✓ LLM - 已添加新LoRA（r={config['lora_llm_r']}, alpha={config['lora_llm_alpha']}）")
        except Exception as e:
            print(f"[Stage2 LoRA] ⚠ LLM - 添加LoRA失败: {e}")
            print(f"[Stage2 LoRA] ⚠ 保持Stage1 LoRA冻结状态")

    # ==================== 步骤3: 启用Quality Decoder训练 ====================
    print("\n[Stage2 LoRA] 步骤3: 启用Quality Decoder训练...")
    if hasattr(model, 'quality_decoder'):
        decoder_params = 0
        for param in model.quality_decoder.parameters():
            param.requires_grad = True
            decoder_params += param.numel()
        print(f"[Stage2 LoRA] ✓ Quality Decoder - 全参数训练（{decoder_params:,} 参数）")

    print("\n[Stage2 LoRA] ========================================")
    print("[Stage2 LoRA] LoRA配置完成")
    print("[Stage2 LoRA] ========================================")

    return model


def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_params = 0

    print("\n[Stage2] 可训练参数列表:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # 只打印关键层
            if 'lora' in name.lower() or 'quality_decoder' in name:
                print(f"  ✓ {name}: {param.numel():,}")

    percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"\n[Stage2] 可训练参数总计: {trainable_params:,} / {all_params:,} ({percentage:.2f}%)")

    return trainable_params, all_params
