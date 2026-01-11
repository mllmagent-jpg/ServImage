"""
Stage2 配置文件 - 基于Stage1权重，使用base模型分类头
"""

import os


def get_stage2_config():
    """
    获取 Stage2 配置

    新的Stage2设计：
    - 加载Stage1的checkpoint（包含base model + LoRA权重）
    - 移除Stage1的CBMQualityDecoder（7维质量评估头）
    - 添加base模型的SingleTaskQualityDecoder（二分类头）
    - 继续训练Stage1的LoRA权重
    - 新的quality_decoder从头训练

    Stage2 LoRA：
    - Vision Encoder LoRA：继续训练Stage1的LoRA
    - Projector LoRA：继续训练Stage1的LoRA
    - Language Model LoRA：继续训练Stage1的LoRA
    - Quality Decoder：新初始化，全参数训练
    """

    # Stage2 LoRA 配置（继续训练Stage1的LoRA）
    config = {
        # Vision Encoder LoRA（继续训练Stage1的LoRA）
        'lora_vision_enable': int(os.getenv('STAGE2_LORA_VISION_ENABLE', '1')) == 1,
        'lora_vision_r': int(os.getenv('STAGE2_LORA_VISION_R', '8')),
        'lora_vision_alpha': int(os.getenv('STAGE2_LORA_VISION_ALPHA', '16')),
        'lora_vision_dropout': float(os.getenv('STAGE2_LORA_VISION_DROPOUT', '0.01')),

        # Projector LoRA（继续训练Stage1的LoRA）
        'lora_projector_enable': int(os.getenv('STAGE2_LORA_PROJECTOR_ENABLE', '1')) == 1,
        'lora_projector_r': int(os.getenv('STAGE2_LORA_PROJECTOR_R', '8')),
        'lora_projector_alpha': int(os.getenv('STAGE2_LORA_PROJECTOR_ALPHA', '16')),
        'lora_projector_dropout': float(os.getenv('STAGE2_LORA_PROJECTOR_DROPOUT', '0.01')),

        # Language Model LoRA（继续训练Stage1的LoRA）
        'lora_llm_enable': int(os.getenv('STAGE2_LORA_LLM_ENABLE', '1')) == 1,
        'lora_llm_r': int(os.getenv('STAGE2_LORA_LLM_R', '8')),
        'lora_llm_alpha': int(os.getenv('STAGE2_LORA_LLM_ALPHA', '16')),
        'lora_llm_dropout': float(os.getenv('STAGE2_LORA_LLM_DROPOUT', '0.01')),

        # Stage2 Decoder 配置（新初始化的SingleTaskQualityDecoder）
        'decoder_dropout': float(os.getenv('STAGE2_DECODER_DROPOUT', '0.1')),
    }

    return config


def get_config():
    """向后兼容的配置获取函数"""
    return get_stage2_config()
