#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to Stage1 model format for Stage2 training

This script loads a trained Stage1 model from a DeepSpeed checkpoint and saves it
in the format expected by Stage2 (with vision_lora, llm_lora, projector_lora, etc.)
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from SerImageModel.models.ServImagemodel_stage1.model_variant4 import Variant4Model
from SerImageModel.models.ServImagemodel_stage1.config_variant4 import get_variant4_config
from transformers import AutoProcessor
import json


def convert_checkpoint(checkpoint_dir: str, output_dir: str):
    """
    Convert DeepSpeed checkpoint to Stage1 model format

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory
        output_dir: Path to output directory for converted model
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"Converting DeepSpeed checkpoint to Stage1 model format")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Find the best model checkpoint
    best_checkpoint = None

    # Check for global_stepXXX directories (created by load_best_model_at_end)
    global_step_dirs = list(checkpoint_path.glob("global_step*"))
    if global_step_dirs:
        # Use the one with highest step number
        best_checkpoint = max(global_step_dirs, key=lambda x: int(x.name.split('step')[1]))
        print(f"Found best model checkpoint: {best_checkpoint}")
    else:
        # Use the latest checkpoint-XXX directory
        checkpoint_dirs = sorted(checkpoint_path.glob("checkpoint-*"),
                                key=lambda x: int(x.name.split('-')[1]))
        if checkpoint_dirs:
            best_checkpoint = checkpoint_dirs[-1]
            print(f"Using latest checkpoint: {best_checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the model from DeepSpeed checkpoint
    print("\n[1/6] Loading model from DeepSpeed checkpoint...")

    # We need to load the base model first, then load the LoRA weights from DeepSpeed
    base_model_path = "weights/Qwen3-VL-2B-Instruct"

    print(f"[1/6] Loading base model: {base_model_path}")
    variant4_config = get_variant4_config()

    model = Variant4Model(
        model_path=base_model_path,
        variant4_config=variant4_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # Load the DeepSpeed checkpoint
    print(f"[1/6] Loading DeepSpeed state dict from: {best_checkpoint}")

    # Use DeepSpeed's zero_to_fp32.py script to consolidate the checkpoint
    zero_to_fp32_script = best_checkpoint / "zero_to_fp32.py"
    if not zero_to_fp32_script.exists():
        # Try parent directory
        zero_to_fp32_script = checkpoint_path / "zero_to_fp32.py"

    if zero_to_fp32_script.exists():
        import subprocess
        consolidated_path = output_path / "consolidated_model.pt"
        print(f"[1/6] Consolidating DeepSpeed checkpoint using zero_to_fp32.py...")
        subprocess.run([
            sys.executable,
            str(zero_to_fp32_script),
            str(best_checkpoint),
            str(consolidated_path)
        ], check=True)

        # Load consolidated checkpoint
        print(f"[1/6] Loading consolidated checkpoint...")
        state_dict = torch.load(consolidated_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

        # Remove consolidated file
        consolidated_path.unlink()
    else:
        print(f"[1/6] ⚠ zero_to_fp32.py not found, trying direct load...")
        # Try to load directly from mp_rank_00_model_states.pt
        model_states_file = best_checkpoint / "global_step500" / "mp_rank_00_model_states.pt"
        if not model_states_file.exists():
            model_states_file = best_checkpoint / "mp_rank_00_model_states.pt"

        if model_states_file.exists():
            print(f"[1/6] Loading model states from: {model_states_file}")
            state_dict = torch.load(model_states_file, map_location='cpu')
            # DeepSpeed saves with 'module.' prefix, need to remove it
            if 'module' in state_dict:
                state_dict = state_dict['module']
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Cannot find model states in {best_checkpoint}")

    print(f"[1/6] ✓ Model loaded successfully")

    # Save Vision LoRA
    print(f"\n[2/6] Saving Vision LoRA...")
    vision_lora_dir = output_path / "vision_lora"
    vision_lora_dir.mkdir(exist_ok=True)
    try:
        model.visual.save_pretrained(str(vision_lora_dir))
        print(f"[2/6] ✓ Vision LoRA saved to: {vision_lora_dir}")
    except Exception as e:
        print(f"[2/6] ⚠ Vision LoRA save failed: {e}")

    # Save Projector LoRA (if exists)
    print(f"\n[3/6] Saving Projector LoRA...")
    if hasattr(model.visual, 'merger'):
        projector_lora_dir = output_path / "projector_lora"
        projector_lora_dir.mkdir(exist_ok=True)
        try:
            model.visual.merger.save_pretrained(str(projector_lora_dir))
            print(f"[3/6] ✓ Projector LoRA saved to: {projector_lora_dir}")
        except Exception as e:
            print(f"[3/6] ⚠ Projector LoRA save failed: {e}")
    else:
        print(f"[3/6] ℹ Model has no projector (visual.merger)")

    # Save LLM LoRA
    print(f"\n[4/6] Saving LLM LoRA...")
    llm_lora_dir = output_path / "llm_lora"
    llm_lora_dir.mkdir(exist_ok=True)
    try:
        model.language_model.save_pretrained(str(llm_lora_dir))
        print(f"[4/6] ✓ LLM LoRA saved to: {llm_lora_dir}")
    except Exception as e:
        print(f"[4/6] ⚠ LLM LoRA save failed: {e}")

    # Save Quality Decoder
    print(f"\n[5/6] Saving Quality Decoder...")
    quality_decoder_dir = output_path / "quality_decoder"
    quality_decoder_dir.mkdir(exist_ok=True)

    quality_decoder_state = {}
    for name, param in model.named_parameters():
        if 'quality_decoder' in name:
            clean_name = name.replace('model.', '') if name.startswith('model.') else name
            quality_decoder_state[clean_name] = param.cpu()

    if quality_decoder_state:
        quality_decoder_path = quality_decoder_dir / "pytorch_model.bin"
        torch.save(quality_decoder_state, quality_decoder_path)
        print(f"[5/6] ✓ Quality Decoder saved: {quality_decoder_path}")
        print(f"[5/6]   Parameters: {len(quality_decoder_state)}")
    else:
        print(f"[5/6] ⚠ No quality_decoder parameters found")

    # Save configs
    print(f"\n[6/6] Saving configurations...")

    # Save model config
    model.config.save_pretrained(str(output_path))
    print(f"[6/6] ✓ Model config saved")

    # Save Variant4 config
    variant4_config_path = output_path / "variant4_config.json"
    with open(variant4_config_path, 'w') as f:
        json.dump(variant4_config, f, indent=2)
    print(f"[6/6] ✓ Variant4 config saved: {variant4_config_path}")

    # Copy processor/tokenizer files
    print(f"[6/6] Copying processor/tokenizer files...")
    processor_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "vocab.json", "merges.txt", "added_tokens.json",
                      "preprocessor_config.json", "video_preprocessor_config.json",
                      "chat_template.jinja"]

    for file_name in processor_files:
        src_file = checkpoint_path / file_name
        if src_file.exists():
            import shutil
            shutil.copy(src_file, output_path / file_name)
            print(f"[6/6]   ✓ Copied: {file_name}")

    print(f"\n{'='*80}")
    print(f"✓ Conversion completed successfully!")
    print(f"{'='*80}")
    print(f"Converted model saved to: {output_path}")
    print(f"\nYou can now use this model for Stage2 training:")
    print(f"  STAGE1_MODEL_PATH={output_path} bash scripts/train_stage2_2binstr.sh")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to Stage1 model format")
    parser.add_argument("checkpoint_dir", type=str, help="Path to DeepSpeed checkpoint directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")

    args = parser.parse_args()

    convert_checkpoint(args.checkpoint_dir, args.output_dir)
