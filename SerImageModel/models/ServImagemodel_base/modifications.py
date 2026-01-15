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


class SingleTaskQualityDecoder(nn.Module):
    """
    单任务质量评估解码器
    与 Variant1 的 SubtaskQualityDecoder 相比，只有一个分类头

    架构（参考LMM4LMM的5层MLP设计）：
    - 输入：使用注意力池化对整个序列进行加权平均
    - 5层MLP金字塔结构：hidden_size -> 1024 -> 256 -> 64 -> 16 -> 2
    - 每层使用ReLU激活函数（而非GELU）
    - 逐步4倍降维，保持平滑的特征转换
    """
    def __init__(self, hidden_size=3584, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 注意力池化层
        self.attention_pooling = AttentionPooling(hidden_size)

        # 5层MLP分类头（参考LMM4LMM设计）
        self.fc1 = nn.Linear(hidden_size, 1024)   # 层1: hidden_size -> 1024
        self.fc2 = nn.Linear(1024, 256)            # 层2: 1024 -> 256
        self.fc3 = nn.Linear(256, 64)              # 层3: 256 -> 64
        self.fc4 = nn.Linear(64, 16)               # 层4: 64 -> 16
        self.fc5 = nn.Linear(16, 2)                # 层5: 16 -> 2 (二分类输出)

        self.relu = nn.ReLU()

        # 添加logits缩放因子，避免输出过小
        self.logits_scale = 100.0  # 将logits放大100倍

        # 可选：在第1层后添加LayerNorm（参考LMM4LMM的infer版本）
        # self.ln1 = nn.LayerNorm(1024)

        # 初始化权重（小范围均匀分布，确保训练稳定性）
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化MLP权重
        修复：使用Kaiming初始化，避免输出过小
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化（He初始化），适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states):
        """
        前向传播

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            logits: (batch_size, 2) - 单个二分类输出
        """
        # 使用注意力池化对整个序列进行加权平均
        pooled_hidden = self.attention_pooling(hidden_states)  # (batch_size, hidden_size)

        # 5层MLP逐步降维
        x = self.relu(self.fc1(pooled_hidden))  # (batch_size, 1024)
        x = self.relu(self.fc2(x))            # (batch_size, 256)
        x = self.relu(self.fc3(x))            # (batch_size, 64)
        x = self.relu(self.fc4(x))            # (batch_size, 16)
        logits = self.fc5(x)                  # (batch_size, 2) - 注意：最后一层不用ReLU

        # 应用缩放因子，增强信号强度
        logits = logits * self.logits_scale

        return logits


def compute_quality_loss(logits, labels):
    """
    计算质量评估损失（简化版，因为只有单个预测）

    Args:
        logits: (batch_size, 2)
        labels: (batch_size,) - 值为0/1

    Returns:
        loss: 标量
    """
    # 类别权重配置 (默认使用1:1平衡权重)
    # 训练集分布: Reject(0): 71.36%, Accept(1): 28.64%
    # 当前设置: [1.0, 1.0] (不进行类别平衡处理)
    # 如需类别平衡，可通过环境变量设置: CLASS_WEIGHT_REJECT, CLASS_WEIGHT_ACCEPT

    # 支持通过环境变量配置权重
    import torch
    import os
    device = logits.device

    # 从环境变量读取权重配置 (默认1:1平衡权重)
    reject_weight = float(os.getenv('CLASS_WEIGHT_REJECT', '1.0'))
    accept_weight = float(os.getenv('CLASS_WEIGHT_ACCEPT', '1.0'))

    class_weights = torch.tensor([reject_weight, accept_weight], dtype=logits.dtype, device=device)

    loss = F.cross_entropy(logits, labels, weight=class_weights)
    return loss


def variant3_forward(
    model,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,  # 这里是单个标签 (batch_size,)
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
    Variant3 的自定义 forward

    整合了：
    1. Qwen3VL 的原始forward（获取hidden states）
    2. SingleTaskQualityDecoder 的预测
    3. 简化的 loss 计算
    """

    # 递归检测
    if hasattr(model, '_in_variant3_forward') and model._in_variant3_forward:
        raise RecursionError("Detected recursive call to variant3_forward! Check forward method wrapping.")

    # 设置递归标志
    model._in_variant3_forward = True

    try:
        # 使用concept_labels（如果提供了）作为labels
        quality_labels = concept_labels if concept_labels is not None else labels

        # 1. 调用原始forward获取输出
        outputs = model._original_forward_func(
            model,  # self
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

        # 关键优化：不返回hidden_states和attentions以节省GPU内存
        # 在评估时，这些中间结果会累积导致OOM
        # 我们已经用它们计算了quality_logits，不再需要保留
        return CausalLMOutputWithPast(
            loss=loss,
            logits=quality_logits,  # 注意：返回质量预测结果
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=None,  # 不返回，避免GPU内存累积
            attentions=None,  # 不返回，避免GPU内存累积
        )
    finally:
        # 清除递归标志
        model._in_variant3_forward = False


def apply_variant3_modifications(model, config=None):
    """
    应用 Variant3 的架构修改

    修改内容：
    1. 添加 SingleTaskQualityDecoder（单头版本）
    2. 重写 forward 方法以支持质量评估
    3. 添加自定义 loss 计算

    Args:
        model: Qwen3VLForConditionalGeneration 实例
        config: 配置字典
    """
    if config is None:
        config = {}

    # 检查是否已经应用过修改
    if hasattr(model, '_variant3_modified') and model._variant3_modified:
        print("[Variant3] 模型已经应用过 Variant3 修改，跳过")
        return model

    # 双重检查：如果 forward 已经是 custom_forward，也跳过
    if hasattr(model.forward, '__name__') and model.forward.__name__ == 'custom_forward':
        print("[Variant3] Forward 方法已被包装，跳过")
        return model

    hidden_size = model.config.text_config.hidden_size
    dropout = config.get('decoder_dropout', 0.1)

    # 1. 添加 Quality Decoder（单头版本）
    model.quality_decoder = SingleTaskQualityDecoder(
        hidden_size=hidden_size,
        dropout=dropout
    )

    # 2. 初始化新参数
    if hasattr(model, '_init_weights'):
        model.quality_decoder.apply(model._init_weights)

    # 3. 保存原始forward方法
    if not hasattr(model, '_original_forward_func'):
        from transformers import Qwen3VLForConditionalGeneration
        model._original_forward_func = Qwen3VLForConditionalGeneration.forward
        print("[Variant3] 已保存原始 forward 方法")

    # 4. 替换forward方法
    def custom_forward(*args, **kwargs):
        return variant3_forward(model, *args, **kwargs)

    model.forward = custom_forward

    # 5. 标记已修改
    model._variant3_modified = True

    print(f"[Variant3] 已添加 SingleTaskQualityDecoder (hidden_size={hidden_size}, dropout={dropout})")
    return model


def apply_lora_to_variant3(model, config=None):
    """
    为 Variant3 应用 LoRA
    与 Variant2 基本相同，但增加了dropout配置

    LoRA 应用位置：
    1. Visual Encoder (model.visual)
    2. Projector (model.visual.merger)
    3. Language Model (model.language_model)
    4. Quality Decoder 不用LoRA，直接全参数训练
    """
    if config is None:
        from .config import get_variant3_config
        config = get_variant3_config()

    vision_lora_enabled = config.get('lora_vision_enable', True)
    projector_lora_enabled = config.get('lora_projector_enable', True)
    llm_lora_enabled = config.get('lora_llm_enable', False)

    # 当 LoRA 完全禁用时
    if not vision_lora_enabled and not projector_lora_enabled and not llm_lora_enabled:
        print("[Variant3 LoRA] 已禁用，冻结基础模型，仅训练 Quality Decoder")
        for name, param in model.named_parameters():
            if 'quality_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    # 1. 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. Vision Encoder LoRA
    if vision_lora_enabled:
        vision_lora_config = LoraConfig(
            r=config.get('lora_vision_r', 8),
            lora_alpha=config.get('lora_vision_alpha', 16),
            lora_dropout=config.get('lora_vision_dropout', 0.05),
            target_modules=["qkv", "proj"],
            bias="none",
        )

        model.visual = get_peft_model(model.visual, vision_lora_config)
        print(f"[Variant3 LoRA] Vision Encoder - r={config['lora_vision_r']}, alpha={config['lora_vision_alpha']}")

    # 3. Projector (visual.merger) LoRA
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
            try:
                model.visual.merger = get_peft_model(model.visual.merger, projector_lora_config)
                print(f"[Variant3 LoRA] Projector LoRA 已应用 - r={config['lora_projector_r']}, alpha={config['lora_projector_alpha']}")
            except Exception as e:
                print(f"[Variant3 LoRA] Projector LoRA 应用失败: {e}")
                print("[Variant3 LoRA] Projector 改为全参数训练")
                for param in model.visual.merger.parameters():
                    param.requires_grad = True
        else:
            print("[Variant3 LoRA] 未找到 visual.merger，跳过 Projector LoRA")

    # 4. Language Model LoRA
    if llm_lora_enabled:
        if not hasattr(model.language_model, "prepare_inputs_for_generation"):
            def _dummy_prepare_inputs_for_generation(*args, **kwargs):
                raise NotImplementedError("Generation is not supported for Variant3 training model.")
            model.language_model.prepare_inputs_for_generation = _dummy_prepare_inputs_for_generation

        llm_lora_config = LoraConfig(
            r=config.get('lora_llm_r', 8),
            lora_alpha=config.get('lora_llm_alpha', 16),
            lora_dropout=config.get('lora_llm_dropout', 0.05),
            target_modules=[
                "q_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.language_model = get_peft_model(model.language_model, llm_lora_config)
        print(f"[Variant3 LoRA] Language Model - r={config['lora_llm_r']}, alpha={config['lora_llm_alpha']}")
    else:
        for param in model.language_model.parameters():
            param.requires_grad = False
        print("[Variant3 LoRA] Language Model - 保持冻结状态")

    # 5. Quality Decoder 全参数训练
    if hasattr(model, 'quality_decoder'):
        for param in model.quality_decoder.parameters():
            param.requires_grad = True
        print("[Variant3 LoRA] Quality Decoder - 全参数训练")

    return model


def print_trainable_parameters(model):
    """
    打印可训练参数统计
    """
    trainable_params = 0
    all_params = 0

    print("\n[Variant3] 可训练参数列表:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # 只打印关键层
            if 'lora' in name.lower() or 'quality_decoder' in name:
                print(f"  ✓ {name}: {param.numel():,}")

    percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"\n[Variant3] 可训练参数总计: {trainable_params:,} / {all_params:,} ({percentage:.2f}%)")

    return trainable_params, all_params
