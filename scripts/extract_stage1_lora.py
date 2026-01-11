import os
import sys
import json
import torch
from pathlib import Path
from collections import OrderedDict


def find_best_checkpoint(stage1_path):
    stage1_path = Path(stage1_path)

    trainer_state_file = stage1_path / "trainer_state.json"
    if trainer_state_file.exists():
        try:
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get('best_model_checkpoint')
                if best_checkpoint and Path(best_checkpoint).exists():
                    print(f"[Extract LoRA] 从 trainer_state.json : {best_checkpoint}")
                    return Path(best_checkpoint)
        except Exception as e:
            print(f"[Extract LoRA] 读取 trainer_state.json 失败: {e}")

    # 2. 查找所有 checkpoint-* 目录
    checkpoints = sorted(stage1_path.glob("checkpoint-*"),
                        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
                        reverse=True)

    if checkpoints:
        print(f"[Extract LoRA] 找到 {len(checkpoints)} 个checkpoint，使用最新的: {checkpoints[0]}")
        return checkpoints[0]

    # 3. 检查根目录是否有 global_step* 目录
    global_steps = sorted(stage1_path.glob("global_step*"),
                         key=lambda x: int(x.name.replace("global_step", "")) if x.name.replace("global_step", "").isdigit() else 0,
                         reverse=True)

    if global_steps:
        print(f"[Extract LoRA] 找到 {len(global_steps)} 个global_step，使用最新的: {global_steps[0]}")
        return global_steps[0]

    return None


def load_deepspeed_checkpoint(checkpoint_dir):
    """加载 DeepSpeed checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)

    # 查找 model_states.pt 文件
    model_states_files = list(checkpoint_dir.glob("**/mp_rank_00_model_states.pt"))

    if not model_states_files:
        raise FileNotFoundError(f"未找到 DeepSpeed checkpoint 文件: {checkpoint_dir}")

    model_states_file = model_states_files[0]
    print(f"[Extract LoRA] 加载 DeepSpeed checkpoint: {model_states_file}")

    # 加载checkpoint
    checkpoint = torch.load(model_states_file, map_location='cpu')

    # DeepSpeed checkpoint 格式: {'module': {...}}
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint

    return state_dict


def extract_lora_weights(state_dict, module_prefix):
    """从完整state_dict中提取指定模块的LoRA权重"""
    lora_weights = OrderedDict()

    for key, value in state_dict.items():
        # 提取包含指定前缀的LoRA参数
        if module_prefix in key and ('lora_' in key or 'modules_to_save' in key):
            # 移除 'module.' 前缀（如果有）
            clean_key = key.replace('module.', '')

            # 移除模块前缀，保留LoRA相对路径
            if module_prefix in clean_key:
                # 保持PEFT格式的键名
                lora_weights[clean_key] = value.cpu()

    return lora_weights


def save_lora_adapter(lora_weights, output_dir, module_type="vision", adapter_name="default"):
    """保存LoRA权重为PEFT格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存权重
    adapter_file = output_dir / "adapter_model.bin"
    torch.save(lora_weights, adapter_file)
    print(f"[Extract LoRA] 权重已保存: {adapter_file} ({len(lora_weights)} 个参数)")

    # 根据模块类型创建完整的 adapter_config.json
    if module_type == "vision":
        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "revision": None,
            "task_type": None,
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["qkv", "proj"],
            "bias": "none",
            "modules_to_save": None,
        }
    elif module_type == "projector":
        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "revision": None,
            "task_type": None,
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["linear_fc1", "linear_fc2"],
            "bias": "none",
            "modules_to_save": None,
        }
    elif module_type == "llm":
        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "revision": None,
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "modules_to_save": None,
        }
    else:
        # 默认配置
        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "revision": None,
            "task_type": None,
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
            "bias": "none",
            "modules_to_save": None,
        }

    config_file = output_dir / "adapter_config.json"
    with open(config_file, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    print(f"[Extract LoRA] 配置已保存: {config_file}")


def extract_quality_decoder(state_dict, output_dir):
    """提取 quality_decoder 权重"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder_weights = OrderedDict()

    for key, value in state_dict.items():
        if 'quality_decoder' in key:
            clean_key = key.replace('module.', '')
            decoder_weights[clean_key] = value.cpu()

    if decoder_weights:
        decoder_file = output_dir / "pytorch_model.bin"
        torch.save(decoder_weights, decoder_file)
        print(f"[Extract LoRA] Quality Decoder已保存: {decoder_file} ({len(decoder_weights)} 个参数)")
        return True

    return False


def check_lora_exists(stage1_path):
    """检查是否已存在LoRA权重"""
    stage1_path = Path(stage1_path)

    # 检查必要的目录
    required_dirs = ["vision_lora", "llm_lora"]
    optional_dirs = ["projector_lora", "quality_decoder"]

    has_required = all((stage1_path / d).exists() for d in required_dirs)

    if has_required:
        print(f"[Extract LoRA] 检测到已存在的LoRA权重:")
        for d in required_dirs + optional_dirs:
            dir_path = stage1_path / d
            if dir_path.exists():
                print(f"  ✓ {d}/")
        return True

    return False


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/extract_stage1_lora.py <stage1_model_path>")
        sys.exit(1)

    stage1_path = Path(sys.argv[1])

    if not stage1_path.exists():
        print(f"[ERROR] Stage1 模型路径不存在: {stage1_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"[Extract LoRA] 开始处理 Stage1 模型: {stage1_path}")
    print(f"{'='*80}\n")

    # 1. 检查是否已有LoRA权重
    if check_lora_exists(stage1_path):
        print(f"\n[Extract LoRA] LoRA权重已存在，跳过提取")
        sys.exit(0)

    print(f"[Extract LoRA] 未找到LoRA权重，开始从DeepSpeed checkpoint提取...\n")

    # 2. 查找最佳checkpoint
    checkpoint_dir = find_best_checkpoint(stage1_path)

    if checkpoint_dir is None:
        print(f"[ERROR] 未找到可用的checkpoint目录")
        sys.exit(1)

    # 3. 加载DeepSpeed checkpoint
    try:
        state_dict = load_deepspeed_checkpoint(checkpoint_dir)
        print(f"[Extract LoRA] Checkpoint加载成功，共 {len(state_dict)} 个参数\n")
    except Exception as e:
        print(f"[ERROR] 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. 提取 Vision LoRA
    print(f"[Extract LoRA] 提取 Vision Encoder LoRA...")
    vision_lora = extract_lora_weights(state_dict, 'visual')
    if vision_lora:
        save_lora_adapter(vision_lora, stage1_path / "vision_lora", module_type="vision")
    else:
        print(f"  ⚠ 未找到Vision LoRA参数")

    # 5. 提取 Projector LoRA
    print(f"\n[Extract LoRA] 提取 Projector LoRA...")
    projector_lora = extract_lora_weights(state_dict, 'merger')
    if projector_lora:
        save_lora_adapter(projector_lora, stage1_path / "projector_lora", module_type="projector")
    else:
        print(f"  ℹ 未找到Projector LoRA参数（可能未训练）")

    # 6. 提取 LLM LoRA
    print(f"\n[Extract LoRA] 提取 Language Model LoRA...")
    llm_lora = extract_lora_weights(state_dict, 'language_model')
    if llm_lora:
        save_lora_adapter(llm_lora, stage1_path / "llm_lora", module_type="llm")
    else:
        print(f"  ⚠ 未找到LLM LoRA参数")

    # 7. 提取 Quality Decoder
    print(f"\n[Extract LoRA] 提取 Quality Decoder...")
    has_decoder = extract_quality_decoder(state_dict, stage1_path / "quality_decoder")
    if not has_decoder:
        print(f"  ℹ 未找到Quality Decoder参数")

    print(f"\n{'='*80}")
    print(f"[Extract LoRA] LoRA权重提取完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
