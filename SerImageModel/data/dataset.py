import json
import torch
import os
import random
import tempfile
import atexit
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataclasses import dataclass
import transformers

"""
ServImage 图像质量评估数据集 - 统一架构

文件组织：
- 第一部分：基础工具函数（所有版本共用）
- 第二部分：V2 - 子任务级评估（展开为独立样本）【已过时，仅保留兼容性】
- 第三部分：V3 - 子任务级评估 + 动态网格图片压缩【已过时，仅保留兼容性】
- 第四部分：V4 - 多维度评分（7个维度详细评分）【已过时，仅保留兼容性】
- 第五部分：统一架构 - 三阶段训练管道（Base/Stage1/Stage2）【推荐使用】
  - 所有任务共享相同的数据加载和动态网格图片压缩逻辑
  - 只在任务prompt和输出格式上有区别
  - Base: Accept/Reject 预测（二分类，0/1标签）- 独立基础模型
  - Stage1: 7维度详细评分（BRF, VEQ-Clarity, VEQ-Realism, VEQ-Aesthetic, VEQ-Text, CNS-Edit, CNS-Set）- 第一阶段训练
  - Stage2: 基于Stage1特征的Accept/Reject判断（7维度评分 + 综合判断）- 第二阶段训练
- 第六部分：便捷创建函数
- 第七部分：使用示例

推荐使用：
- 新项目：使用统一架构（Base/Stage1/Stage2），通过 task_mode 参数指定任务类型
- 旧项目：继续使用 V2/V3/V4（已过时），通过 version 参数指定版本

图片压缩策略（统一架构和V3/V4）：
- 当前结果图：1张原图（不压缩）
- 输入图：最多1张原图（不压缩）
- 其它结果图：压缩为最多2张拼贴图（动态网格：2x2, 2x3, 3x3等）
- 参考图：压缩为最多2张拼贴图（动态网格：2x2, 2x3, 3x3等）
- 总图片数：最多6张
- 动态网格：根据图片数量自动计算最优网格布局
"""

# ============================================================================
# 第一部分：基础工具函数（所有版本共用）
# ============================================================================

def get_images_from_folder(folder_path: str) -> List[str]:
    """
    获取文件夹下的所有图片路径

    Args:
        folder_path: 文件夹路径

    Returns:
        图片路径列表（排序后）
    """
    folder = Path(folder_path)
    if not folder.exists():
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in image_extensions:
            images.append(str(file))

    return images


def load_bench_tasks(bench_dir: str) -> Dict[str, Any]:
    """
    加载 ServImage_Bench 中的所有任务定义

    Args:
        bench_dir: ServImage_Bench 目录路径

    Returns:
        {task_id: task_data, ...}
    """
    bench_path = Path(bench_dir)
    tasks = {}

    bench_files = [
        (bench_path / "C-Portrait" / "Portrait.json", "C-Portrait"),
        (bench_path / "C-Digital" / "Digital.json", "C-Digital"),
        (bench_path / "C-Product" / "Product.json", "C-Product"),
    ]

    for filepath, category in bench_files:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                task_list = json.load(f)
                for task in task_list:
                    task_id = task['task_id']
                    task['_category'] = category
                    tasks[task_id] = task

                    # 创建 task_XXX 格式的别名以兼容训练数据格式
                    # 例如: product-001 -> task_001, digital-123 -> task_123
                    if '-' in task_id:
                        try:
                            task_num = task_id.split('-')[-1]
                            alias_id = f"task_{task_num}"
                            tasks[alias_id] = task
                        except (ValueError, IndexError):
                            pass  # 如果格式不符合预期，跳过别名创建
            print(f"[Dataset] 加载 {filepath.name}: {len(task_list)} 个任务")
        else:
            print(f"[Dataset] 警告: 文件不存在 - {filepath}")

    print(f"[Dataset] 总共加载 {len(tasks)} 个任务定义（包含别名）")
    return tasks


def task_id_to_folder_name(task_id: str, category: str) -> str:
    """
    将 task_id 转换为文件夹名称

    例如: "digital-016" + "C-Digital" -> "C-Digital_task_016"
          "task_353" + "C-Product" -> "C-Product_task_353"
    """
    # 处理两种格式：
    # 1. 旧格式: "digital-016" -> 取最后部分的数字
    # 2. 新格式: "task_353" -> 移除 "task_" 前缀
    if task_id.startswith('task_'):
        # 新格式：task_XXX
        task_num_str = task_id.replace('task_', '')
        task_num = int(task_num_str)
    else:
        # 旧格式：xxx-YYY
        task_num = int(task_id.split('-')[-1])
    return f"{category}_task_{task_num:03d}"


def get_task_folder_images(bench_dir: Path, category: str, subfolder: str, task_num: int) -> List[str]:
    """
    获取任务文件夹中的图片，尝试两种命名格式（带前导零和不带前导零）

    Args:
        bench_dir: Bench 目录路径
        category: 类别名称（如 C-Digital）
        subfolder: 子文件夹名称（inputs 或 refs）
        task_num: 任务编号（整数）

    Returns:
        图片路径列表
    """
    # 尝试带前导零的格式（task_001, task_010, task_100）
    folder_with_zero = bench_dir / category / subfolder / f"task_{task_num:03d}"
    images = get_images_from_folder(folder_with_zero)
    if images:
        return images

    # 尝试不带前导零的格式（task_1, task_10, task_100）
    folder_without_zero = bench_dir / category / subfolder / f"task_{task_num}"
    images = get_images_from_folder(folder_without_zero)
    return images



# ============================================================================
# 第二部分：V2 - 子任务级评估（已过时，仅保留兼容性）
# ============================================================================
# 说明：推荐使用第五部分的统一架构（Base/Stage1/Stage2）
# ============================================================================

def build_system_prompt_v2() -> str:
    """
    构建系统提示词（V2版本）

    特点：明确说明只评估第一张（current）图片
    """
    prompt = """You are a professional image processing evaluation assistant.

I will provide you with:
- A task description with specific requirements and subtasks
- The CURRENT IMAGE that you need to evaluate (marked as "current_result_image")
- Other result images for context (these are NOT the images you need to evaluate)
- Task input images (if this is an image modification task)
- Reference images

Your task is to evaluate ONLY the current image (the first result image) to determine if it meets the user's requirements. Focus on whether this single image satisfies the task requirements.

Output your evaluation as a binary decision: accept (1) or reject (0)."""

    return prompt


def build_qwen3vl_message_v2(
    role_prompt: str,
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    shuffle_context: bool = True,
) -> Dict[str, Any]:
    """
    构建Qwen3-VL的消息格式（V2版本）

    特点：
    - 明确标注第一张图片是当前要评估的
    - 其他图片作为上下文
    - 支持随机打乱上下文图片顺序

    Args:
        role_prompt: 角色提示词
        task_data: 任务JSON数据
        current_result_image: 当前要评估的结果图片路径
        other_result_images: 其他结果图片路径列表
        input_images: 任务输入图片路径列表
        ref_images: 任务参考图片路径列表
        shuffle_context: 是否随机打乱上下文图片

    Returns:
        包含消息列表和图片列表的字典
    """
    # 提取任务信息
    original_data = task_data.get("original_task_data", {})
    requirements = original_data.get("requirements_original", "")
    output_quantity = original_data.get("output_quantity", 0)
    price_usd = original_data.get("price_usd", 0)

    # 提取子任务
    deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
    subtasks = []
    for i, d in enumerate(deliverables, 1):
        subtask_text = d.get("subtask", "")
        if subtask_text:
            subtasks.append(f"subtask_{i}: {subtask_text}")

    # 构建文本内容
    text_parts = []

    # 1. 任务需求
    text_parts.append(f"task_requirement: {requirements}")

    # 2. 子任务列表
    if subtasks:
        text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
        text_parts.append("\n".join(subtasks))

    # 3. 数字信息
    text_parts.append(f"\n\noutput_quantity: {output_quantity}")
    text_parts.append(f"task_value: ${price_usd}")

    # 4. 图片部分 - 关键变化：说明第一张是待评估图片
    text_parts.append(f"\n\n## 图片")

    # 当前评估图片（重点）
    text_parts.append(f"\n### 当前评估图片 (Current Image to Evaluate)")
    text_parts.append(f"current_result_image: <image>")
    text_parts.append(f"Note: This is the image you need to evaluate against the task requirements.")

    # 其他结果图片（上下文）
    if other_result_images:
        # 数据增强：随机打乱上下文图片顺序
        context_images = other_result_images.copy()
        if shuffle_context:
            random.shuffle(context_images)

        text_parts.append(f"\n### 其他结果图片 (Other Result Images for Context, {len(context_images)}张)")
        text_parts.append(f"Note: These are other generated images for reference.")
        for i in range(len(context_images)):
            text_parts.append(f"context_result_{i+1}: <image>")

    # 任务输入图片
    if input_images:
        text_parts.append(f"\n### 任务输入图片 ({len(input_images)}张)")
        for i in range(len(input_images)):
            text_parts.append(f"input_image_{i+1}: <image>")

    # 任务参考图片
    if ref_images:
        text_parts.append(f"\n### 任务参考图片 ({len(ref_images)}张)")
        for i in range(len(ref_images)):
            text_parts.append(f"ref_image_{i+1}: <image>")

    user_text = "\n".join(text_parts)

    # 构建消息列表
    messages = []

    # 系统消息
    if role_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": role_prompt}]
        })

    # 用户消息
    user_content = []

    # 图片顺序：当前图片 → 其他结果图片 → 输入图片 → 参考图片
    all_images = [current_result_image]
    if other_result_images:
        if shuffle_context:
            context_images = other_result_images.copy()
            random.shuffle(context_images)
            all_images.extend(context_images)
        else:
            all_images.extend(other_result_images)
    all_images.extend(input_images)
    all_images.extend(ref_images)

    user_content.append({"type": "text", "text": user_text})

    # 图片content项
    for img_path in all_images:
        user_content.append({"type": "image", "image": img_path})

    messages.append({
        "role": "user",
        "content": user_content
    })

    return {
        "messages": messages,
        "images": all_images,
        "metadata": {
            "task_id": task_data.get("task_id", ""),
            "num_input_images": len(input_images),
            "num_ref_images": len(ref_images),
            "num_result_images": 1 + len(other_result_images),
            "total_images": len(all_images)
        }
    }


def build_quality_eval_sample_v2(
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    label: int,
    shuffle_context: bool = True,
) -> Dict[str, Any]:
    """
    构建单个训练样本（V2版本）

    Returns:
        {
            "messages": [...],
            "images": [...],
            "label": int,
            "task_id": str
        }
    """
    system_prompt = build_system_prompt_v2()

    message_data = build_qwen3vl_message_v2(
        role_prompt=system_prompt,
        task_data=task_data,
        current_result_image=current_result_image,
        other_result_images=other_result_images,
        input_images=input_images,
        ref_images=ref_images,
        shuffle_context=shuffle_context
    )

    return {
        **message_data,
        "label": label,
        "task_id": task_data.get("task_id", "")
    }


class QualityEvalDatasetV2(Dataset):
    """
    图像质量评估数据集（V2版本）

    特点：
    - 每个子任务生成一个独立样本
    - 样本包含：当前评估图片 + 其他图片作为上下文
    - 标签是单个值（0/1）而不是数组
    """
    def __init__(
        self,
        train_jsonl_path: str,
        bench_dir: str,
        dataset_dir: str,
        shuffle_context: bool = True,
        processor=None
    ):
        """
        Args:
            train_jsonl_path: 训练数据 jsonl 文件路径
            bench_dir: ServImage_Bench 目录路径
            dataset_dir: ServImage_Dataset 目录路径
            shuffle_context: 是否随机打乱上下文图片
            processor: Qwen3VL 的 processor
        """
        self.shuffle_context = shuffle_context
        self.processor = processor
        self.bench_dir = Path(bench_dir)
        self.dataset_dir = Path(dataset_dir)

        # 加载任务定义
        print(f"[QualityDatasetV2] 加载任务定义: {bench_dir}")
        self.bench_tasks = load_bench_tasks(bench_dir)

        # 加载训练数据并展开为子任务样本
        print(f"[QualityDatasetV2] 加载训练数据: {train_jsonl_path}")
        self.samples = []
        self._load_and_expand_samples(train_jsonl_path)

        print(f"[QualityDatasetV2] 总共加载 {len(self.samples)} 个训练样本（已展开为子任务）")

    def _load_and_expand_samples(self, jsonl_path: str):
        """
        从 jsonl 文件加载数据并展开为子任务样本

        关键：一个任务如果有N个子任务，则生成N个独立样本
        """
        loaded_tasks = 0
        loaded_samples = 0
        skipped_tasks = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    model = data['model']
                    task_id = data['task_id']
                    category = data['category']
                    labels = data['labels']
                    num_valid_subtasks = data['num_valid_subtasks']

                    # 检查任务定义
                    if task_id not in self.bench_tasks:
                        print(f"[QualityDatasetV2] 警告: 任务 {task_id} 未在 Bench 中找到，跳过")
                        skipped_tasks += 1
                        continue

                    task_data = self.bench_tasks[task_id]

                    # 构建图片路径
                    folder_name = task_id_to_folder_name(task_id, category)

                    result_images = get_images_from_folder(
                        self.dataset_dir / model / folder_name
                    )

                    task_num = int(task_id.split('-')[-1])

                    input_images = get_task_folder_images(
                        self.bench_dir, category, "inputs", task_num
                    )

                    ref_images = get_task_folder_images(
                        self.bench_dir, category, "refs", task_num
                    )

                    if not result_images:
                        skipped_tasks += 1
                        continue

                    # 关键：展开为子任务样本
                    for subtask_idx in range(num_valid_subtasks):
                        label = labels[subtask_idx]

                        if label == -100:
                            continue

                        if subtask_idx < len(result_images):
                            current_image = result_images[subtask_idx]
                            other_images = (
                                result_images[:subtask_idx] +
                                result_images[subtask_idx+1:]
                            )

                            sample = self._build_sample(
                                task_data=task_data,
                                current_image=current_image,
                                other_images=other_images,
                                input_images=input_images,
                                ref_images=ref_images,
                                label=label,
                                model=model,
                                task_id=task_id,
                                subtask_idx=subtask_idx
                            )

                            self.samples.append(sample)
                            loaded_samples += 1

                    loaded_tasks += 1

                except Exception as e:
                    print(f"[QualityDatasetV2] 错误: 第 {line_num} 行处理失败 - {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_tasks += 1

        print(f"[QualityDatasetV2] 处理任务: {loaded_tasks}, 生成样本: {loaded_samples}, 跳过: {skipped_tasks}")

    def _build_sample(
        self,
        task_data: Dict[str, Any],
        current_image: str,
        other_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        label: int,
        model: str,
        task_id: str,
        subtask_idx: int
    ) -> Dict[str, Any]:
        """构建单个训练样本"""
        sample = build_quality_eval_sample_v2(
            task_data=task_data,
            current_result_image=current_image,
            other_result_images=other_images,
            input_images=input_images,
            ref_images=ref_images,
            label=label,
            shuffle_context=self.shuffle_context
        )

        # 添加元信息
        sample['model'] = model
        sample['task_id'] = task_id
        sample['subtask_idx'] = subtask_idx

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            处理后的字典，包含 input_ids, pixel_values, attention_mask, label 等
        """
        sample = self.samples[idx]

        if self.processor:
            try:
                messages = sample['messages']

                processed = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False
                )

                inputs = {}
                for k, v in processed.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() > 1 and v.size(0) == 1:
                            inputs[k] = v.squeeze(0).contiguous()
                        else:
                            inputs[k] = v.contiguous()
                    else:
                        inputs[k] = v

                inputs['labels'] = torch.tensor(sample['label'], dtype=torch.long)

                return inputs
            except Exception as e:
                print(f"\n[QualityDatasetV2 ERROR] 处理样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to process sample {idx}: {e}")

        return sample


@dataclass
class QualityEvalCollatorV2:
    """
    图像质量评估任务的自定义 Collator（V2版本）

    特点：
    - labels 是单个值数组 (batch,) 而不是 (batch, max_subtasks)
    - 更简单的批处理逻辑
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """批处理多个样本"""
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        labels_list = []

        for feature in features:
            # 文本部分
            if 'input_ids' in feature:
                ids = feature['input_ids']
                if isinstance(ids, torch.Tensor):
                    input_ids_list.append(ids)
                else:
                    input_ids_list.append(torch.tensor(ids, dtype=torch.long))

            if 'attention_mask' in feature:
                mask = feature['attention_mask']
                if isinstance(mask, torch.Tensor):
                    attention_mask_list.append(mask)
                else:
                    attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

            # 图像部分
            if 'pixel_values' in feature:
                pv = feature['pixel_values']
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pixel_values_list.append(pv)

            if 'image_grid_thw' in feature:
                thw = feature['image_grid_thw']
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                image_grid_thw_list.append(thw)

            # 标签部分（单个值）
            if 'labels' in feature:
                label = feature['labels']
                if isinstance(label, torch.Tensor):
                    if label.dim() == 0:
                        labels_list.append(label)
                    else:
                        labels_list.append(label.squeeze())
                else:
                    labels_list.append(torch.tensor(label, dtype=torch.long))

        # 构建批次字典
        batch = {}

        # Pad input_ids 和 attention_mask
        if input_ids_list:
            max_length = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(ids)

                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_ids)

                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)

            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_mask)

        # Stack pixel_values
        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        # Stack image_grid_thw
        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        # Stack labels
        if labels_list:
            batch['labels'] = torch.stack(labels_list)

        return batch


# ============================================================================
# 第三部分：V3 - 子任务级评估 + 动态网格图片压缩（已过时，仅保留兼容性）
# ============================================================================
# 说明：V3 现已使用动态网格压缩策略（根据图片数量自动计算网格布局）
#      推荐使用第五部分的统一架构（Base/Stage1/Stage2）
# ============================================================================

def make_composite(
    images: List[str],
    size: tuple = (224, 224),
    grid: tuple = (2, 2)
) -> Optional[Image.Image]:
    """
    将多张图片拼贴成一张网格图

    Args:
        images: 图片路径列表
        size: 输出图片尺寸 (width, height)
        grid: 网格布局 (cols, rows)

    Returns:
        PIL.Image 对象，如果输入为空则返回 None
    """
    if not images:
        return None

    W, H = size
    cols, rows = grid
    tile_w, tile_h = W // cols, H // rows

    canvas = Image.new("RGB", size, color=(0, 0, 0))

    for idx, img_path in enumerate(images):
        if idx >= cols * rows:
            break

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((tile_w, tile_h))

            row = idx // cols
            col = idx % cols
            x0, y0 = col * tile_w, row * tile_h

            canvas.paste(img, (x0, y0))
        except Exception as e:
            print(f"[Warning] Failed to load image {img_path}: {e}")
            continue

    return canvas


def save_temp_image(img: Image.Image, prefix: str = "composite") -> str:
    """
    将PIL图像保存到临时文件

    Args:
        img: PIL.Image 对象
        prefix: 临时文件名前缀

    Returns:
        临时文件路径
    """
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png",
        prefix=f"{prefix}_"
    )
    img.save(temp_file.name, format="PNG")
    temp_file.close()
    return temp_file.name


def save_image_to_path(img: Image.Image, file_path: str) -> str:
    """
    将PIL图像保存到指定路径

    Args:
        img: PIL.Image 对象
        file_path: 目标文件路径

    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    img.save(file_path, format="PNG")
    return file_path


def get_composite_cache_path(
    images: List[str],
    composite_idx: int,
    cache_dir: str = "data/train/compressed_images"
) -> str:
    """
    根据原始图片列表生成缓存文件路径

    使用图片路径列表的哈希值作为唯一标识

    Args:
        images: 原始图片路径列表（已排序）
        composite_idx: 拼贴图索引（0, 1, 2, ...）
        cache_dir: 缓存目录

    Returns:
        缓存文件路径
    """
    # 生成唯一标识：基于图片列表的哈希
    images_str = "|".join(sorted(images))
    hash_obj = hashlib.md5(images_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()

    # 文件名格式：hash_composite_idx.png
    filename = f"{hash_hex}_comp_{composite_idx}.png"
    cache_path = os.path.join(cache_dir, filename)

    return cache_path


def compress_images_to_composites(
    images: List[str],
    max_composites: int = 2,
    grid_size: tuple = None,
    canvas_size: tuple = (224, 224),
    use_cache: bool = True,
    cache_dir: str = "data/train/compressed_images"
) -> List[str]:
    """
    将图片列表压缩成最多 N 张拼贴图（动态网格版本，支持缓存）

    改进：
    - 不再丢弃超出固定网格的图片，而是自动调整网格大小
    - 支持缓存压缩结果，减少重复压缩开销

    Args:
        images: 原始图片路径列表
        max_composites: 最多生成多少张拼贴图
        grid_size: 已废弃（保留参数兼容性）
        canvas_size: 每张拼贴图的画布尺寸
        use_cache: 是否使用缓存
        cache_dir: 缓存目录

    Returns:
        拼贴图路径列表（可能是缓存路径或临时文件路径）
    """
    if not images:
        return []

    print(f"[ImageCompress] 压缩 {len(images)} 张图片 → 最多 {max_composites} 张拼贴图 (cache={use_cache})")

    sorted_images = sorted(images)
    K = len(sorted_images)

    composite_paths = []
    images_per_composite = (K + max_composites - 1) // max_composites

    for i in range(max_composites):
        start_idx = i * images_per_composite
        end_idx = min(start_idx + images_per_composite, K)

        group = sorted_images[start_idx:end_idx]
        if not group:
            break

        # 检查缓存
        if use_cache:
            cache_path = get_composite_cache_path(group, i, cache_dir)

            if os.path.exists(cache_path):
                print(f"[ImageCompress] 使用缓存: {os.path.basename(cache_path)}")
                composite_paths.append(cache_path)
                continue

        # 动态计算网格大小
        M = len(group)
        import math
        cols = math.ceil(math.sqrt(M))
        rows = math.ceil(M / cols)
        dynamic_grid = (cols, rows)

        composite_img = make_composite(group, size=canvas_size, grid=dynamic_grid)
        if composite_img:
            # 保存到缓存或临时文件
            if use_cache:
                cache_path = get_composite_cache_path(group, i, cache_dir)
                composite_path = save_image_to_path(composite_img, cache_path)
                print(f"[ImageCompress] 已缓存: {os.path.basename(cache_path)} ({len(group)} images, {cols}x{rows} grid)")
            else:
                composite_path = save_temp_image(composite_img, prefix=f"composite_{i+1}")

            composite_paths.append(composite_path)

    return composite_paths


def build_system_prompt_v3() -> str:
    """
    构建系统提示词（V3版本）

    强调图片可能以动态网格形式呈现
    """
    prompt = """You are a professional image processing evaluation assistant.

I will provide you with:
- A task description with specific requirements and subtasks
- The CURRENT IMAGE that you need to evaluate (marked as "current_result_image")
- Other result images for context (may be shown as grid composites with varying layouts)
- Task input images (if this is an image modification task)
- Reference images (may be shown as grid composites with varying layouts)

Your task is to evaluate ONLY the current image to determine if it meets the specific requirement for the indicated subtask. Focus on whether this single image satisfies that particular subtask requirement.

Note: Some images may be presented as grid composites containing multiple images (e.g., 2x1, 2x2, 3x2, 3x3, 4x3, etc.) for efficient visual token usage. The grid layout is automatically adjusted based on the number of images. You can still evaluate the current image by referring to these composite images.

Output your evaluation as a binary decision: accept (1) or reject (0)."""

    return prompt


def build_qwen3vl_message_v3(
    role_prompt: str,
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    subtask_idx: int,
    shuffle_context: bool = False,
) -> Dict[str, Any]:
    """
    构建Qwen3-VL的消息格式（V3版本）

    关键改进：
    - 当前结果图：1张，原样输入
    - 输入图：最多1张，原样输入
    - 其它结果图：压缩成最多2张拼贴图
    - 参考图：压缩成最多2张拼贴图
    - 总图片数：最多6张
    - 明确指出当前评估的子任务
    - 返回临时文件列表用于后续清理

    Args:
        role_prompt: 角色提示词
        task_data: 任务JSON数据
        current_result_image: 当前要评估的结果图片路径
        other_result_images: 其他结果图片路径列表
        input_images: 任务输入图片路径列表
        ref_images: 任务参考图片路径列表
        subtask_idx: 当前评估的子任务索引（0-based）
        shuffle_context: 是否随机打乱

    Returns:
        包含消息列表、图片列表和临时文件列表的字典
    """
    # 提取任务信息
    original_data = task_data.get("original_task_data", {})
    requirements = original_data.get("requirements_original", "")
    output_quantity = original_data.get("output_quantity", 0)
    price_usd = original_data.get("price_usd", 0)

    # 提取子任务
    deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
    subtasks = []
    for i, d in enumerate(deliverables, 1):
        subtask_text = d.get("subtask", "")
        if subtask_text:
            subtasks.append(f"subtask_{i}: {subtask_text}")

    # ========== V3 核心：图片压缩策略 ==========
    temp_files = []

    # 1. 当前评估图片
    images_to_use = [current_result_image]

    # 2. 输入图片：最多1张
    input_images_to_use = input_images[:1] if input_images else []

    # 3. 其它结果图片：压缩成最多2张拼贴图（动态网格）
    other_result_composites = compress_images_to_composites(
        other_result_images,
        max_composites=2,
        canvas_size=(224, 224),
        use_cache=False  # V3使用临时文件，不启用缓存
    )
    temp_files.extend(other_result_composites)

    # 4. 参考图片：压缩成最多2张拼贴图（动态网格）
    ref_composites = compress_images_to_composites(
        ref_images,
        max_composites=2,
        canvas_size=(224, 224),
        use_cache=False  # V3使用临时文件，不启用缓存
    )
    temp_files.extend(ref_composites)

    # 构建最终图片列表
    images_to_use.extend(input_images_to_use)
    images_to_use.extend(other_result_composites)
    images_to_use.extend(ref_composites)

    # ========== 构建文本内容 ==========
    text_parts = []

    # 1. 任务需求
    text_parts.append(f"task_requirement: {requirements}")

    # 2. 子任务列表
    if subtasks:
        text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
        text_parts.append("\n".join(subtasks))

    # 3. 数字信息
    text_parts.append(f"\n\noutput_quantity: {output_quantity}")
    text_parts.append(f"task_value: ${price_usd}")

    # 4. 图片部分
    text_parts.append(f"\n\n## 图片")

    # 当前评估图片 - 明确说明当前评估的子任务
    text_parts.append(f"\n### 当前评估图片 (Current Image to Evaluate)")

    if subtask_idx < len(subtasks):
        current_subtask = subtasks[subtask_idx]
        if ": " in current_subtask:
            subtask_name, subtask_desc = current_subtask.split(": ", 1)
        else:
            subtask_name = current_subtask
            subtask_desc = ""

        text_parts.append(f"Evaluating: {subtask_name}")
        if subtask_desc:
            text_parts.append(f'Requirement: "{subtask_desc}"')
    else:
        subtask_name = f"subtask_{subtask_idx + 1}"
        text_parts.append(f"Evaluating: {subtask_name}")

    text_parts.append(f"current_result_image: <image>")
    text_parts.append(
        f"Note: Please evaluate whether THIS image satisfies the specific requirement "
        f"for {subtask_name if subtask_idx < len(subtasks) else f'subtask_{subtask_idx + 1}'}. "
        f"Focus on this particular subtask when making your accept/reject decision."
    )

    # 输入图片
    if input_images_to_use:
        text_parts.append(f"\n### 任务输入图片 ({len(input_images_to_use)}张)")
        for i in range(len(input_images_to_use)):
            text_parts.append(f"input_image_{i+1}: <image>")

    # 其它结果图片（压缩版）
    if other_result_composites:
        num_original = len(other_result_images)
        text_parts.append(
            f"\n### 其他结果图片 (Compressed from {num_original} images into {len(other_result_composites)} composite(s))"
        )
        text_parts.append(f"Note: These are other generated images shown in grid layouts for reference.")
        for i in range(len(other_result_composites)):
            text_parts.append(f"other_results_composite_{i+1}: <image>")

    # 参考图片（压缩版）
    if ref_composites:
        num_original = len(ref_images)
        text_parts.append(
            f"\n### 任务参考图片 (Compressed from {num_original} images into {len(ref_composites)} composite(s))"
        )
        text_parts.append(f"Note: These are reference images shown in grid layouts.")
        for i in range(len(ref_composites)):
            text_parts.append(f"ref_composite_{i+1}: <image>")

    user_text = "\n".join(text_parts)

    # ========== 构建消息列表 ==========
    messages = []

    if role_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": role_prompt}]
        })

    user_content = []
    user_content.append({"type": "text", "text": user_text})

    for img_path in images_to_use:
        user_content.append({"type": "image", "image": img_path})

    messages.append({
        "role": "user",
        "content": user_content
    })

    return {
        "messages": messages,
        "images": images_to_use,
        "temp_files": temp_files,
        "metadata": {
            "task_id": task_data.get("task_id", ""),
            "num_input_images": len(input_images_to_use),
            "num_ref_images_original": len(ref_images),
            "num_ref_composites": len(ref_composites),
            "num_other_results_original": len(other_result_images),
            "num_other_results_composites": len(other_result_composites),
            "total_images": len(images_to_use)
        }
    }


def build_quality_eval_sample_v3(
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    label: int,
    subtask_idx: int,
    shuffle_context: bool = False,
) -> Dict[str, Any]:
    """
    构建单个训练样本（V3版本）

    Args:
        task_data: 任务数据字典
        current_result_image: 当前评估的图片路径
        other_result_images: 其他结果图片路径列表
        input_images: 输入图片路径列表
        ref_images: 参考图片路径列表
        label: 标签（0=拒绝, 1=接受）
        subtask_idx: 当前评估的子任务索引
        shuffle_context: 是否随机打乱上下文

    Returns:
        {
            "messages": [...],
            "images": [...],
            "label": int,
            "task_id": str,
            "temp_files": [...]
        }
    """
    system_prompt = build_system_prompt_v3()

    message_data = build_qwen3vl_message_v3(
        role_prompt=system_prompt,
        task_data=task_data,
        current_result_image=current_result_image,
        other_result_images=other_result_images,
        input_images=input_images,
        ref_images=ref_images,
        subtask_idx=subtask_idx,
        shuffle_context=shuffle_context
    )

    return {
        **message_data,
        "label": label,
        "task_id": task_data.get("task_id", "")
    }


class QualityEvalDatasetV3(Dataset):
    """
    图像质量评估数据集（V3版本）

    关键改进：
    - 每个子任务生成一个独立样本
    - 使用图片压缩策略减少视觉token
    - 添加临时文件管理
    - 使用缓存机制避免重复生成composite图片
    """
    def __init__(
        self,
        train_jsonl_path: str,
        bench_dir: str,
        dataset_dir: str,
        shuffle_context: bool = False,
        processor=None
    ):
        """
        Args:
            train_jsonl_path: 训练数据 jsonl 文件路径
            bench_dir: ServImage_Bench 目录路径
            dataset_dir: ServImage_Dataset 目录路径
            shuffle_context: 是否随机打乱上下文图片
            processor: Qwen3VL 的 processor
        """
        self.shuffle_context = shuffle_context
        self.processor = processor
        self.bench_dir = Path(bench_dir)
        self.dataset_dir = Path(dataset_dir)

        # 临时文件管理
        self._temp_files = set()
        self._composite_cache = {}
        atexit.register(self._cleanup_temp_files)

        # 加载任务定义
        print(f"[QualityDatasetV3] 加载任务定义: {bench_dir}")
        self.bench_tasks = load_bench_tasks(bench_dir)

        # 加载训练数据并展开为子任务样本
        print(f"[QualityDatasetV3] 加载训练数据: {train_jsonl_path}")
        self.samples = []
        self._load_and_expand_samples(train_jsonl_path)

        print(f"[QualityDatasetV3] 总共加载 {len(self.samples)} 个训练样本（已展开为子任务）")

    def _cleanup_temp_files(self):
        """清理所有生成的临时文件"""
        if not self._temp_files:
            return

        print(f"[QualityDatasetV3] 清理 {len(self._temp_files)} 个临时文件...")
        cleaned = 0
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    cleaned += 1
            except Exception as e:
                print(f"[QualityDatasetV3] 警告: 无法删除临时文件 {temp_file}: {e}")

        print(f"[QualityDatasetV3] 已清理 {cleaned}/{len(self._temp_files)} 个临时文件")
        self._temp_files.clear()
        self._composite_cache.clear()

    def __del__(self):
        """析构函数：确保对象销毁时清理临时文件"""
        self._cleanup_temp_files()

    def _load_and_expand_samples(self, jsonl_path: str):
        """
        从 jsonl 文件加载数据并展开为子任务样本

        支持两种数据格式：
        1. 任务级格式（旧）：包含 labels 数组和 num_valid_subtasks
        2. 子任务级格式（新）：每行就是一个子任务
        """
        loaded_tasks = 0
        loaded_samples = 0
        skipped_tasks = 0

        # 统计总行数
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())

        print(f"[QualityDatasetV3] 开始加载 {total_lines} 行数据...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line_num % 1000 == 0:
                    print(f"[QualityDatasetV3] 进度: {line_num}/{total_lines}, 已生成 {loaded_samples} 个样本")

                try:
                    data = json.loads(line)

                    model = data['model']
                    task_id = data['task_id']
                    category = data['category']

                    if task_id not in self.bench_tasks:
                        print(f"[QualityDatasetV3] 警告: 任务 {task_id} 未在 Bench 中找到，跳过")
                        skipped_tasks += 1
                        continue

                    task_data = self.bench_tasks[task_id]

                    # 构建图片路径
                    folder_name = task_id_to_folder_name(task_id, category)

                    result_images = get_images_from_folder(
                        self.dataset_dir / model / folder_name
                    )

                    task_num = int(task_id.split('-')[-1])

                    input_images = get_task_folder_images(
                        self.bench_dir, category, "inputs", task_num
                    )

                    ref_images = get_task_folder_images(
                        self.bench_dir, category, "refs", task_num
                    )

                    if not result_images:
                        skipped_tasks += 1
                        continue

                    # 判断数据格式
                    if 'labels' in data and 'num_valid_subtasks' in data:
                        # 任务级格式：展开
                        labels = data['labels']
                        num_valid_subtasks = data['num_valid_subtasks']

                        for subtask_idx in range(num_valid_subtasks):
                            label = labels[subtask_idx]

                            if label == -100:
                                continue

                            if subtask_idx < len(result_images):
                                current_image = result_images[subtask_idx]
                                other_images = (
                                    result_images[:subtask_idx] +
                                    result_images[subtask_idx+1:]
                                )

                                sample = self._build_sample(
                                    task_data=task_data,
                                    current_image=current_image,
                                    other_images=other_images,
                                    input_images=input_images,
                                    ref_images=ref_images,
                                    label=label,
                                    subtask_idx=subtask_idx,
                                    model=model,
                                    task_id=task_id
                                )

                                self.samples.append(sample)
                                loaded_samples += 1

                        loaded_tasks += 1

                    elif 'label' in data and 'subtask_idx' in data:
                        # 子任务级格式：直接使用
                        label = data['label']
                        subtask_idx = data['subtask_idx']

                        if subtask_idx < len(result_images):
                            current_image = result_images[subtask_idx]
                            other_images = (
                                result_images[:subtask_idx] +
                                result_images[subtask_idx+1:]
                            )

                            sample = self._build_sample(
                                task_data=task_data,
                                current_image=current_image,
                                other_images=other_images,
                                input_images=input_images,
                                ref_images=ref_images,
                                label=label,
                                subtask_idx=subtask_idx,
                                model=model,
                                task_id=task_id
                            )

                            self.samples.append(sample)
                            loaded_samples += 1

                except Exception as e:
                    print(f"[QualityDatasetV3] 错误: 第 {line_num} 行处理失败 - {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_tasks += 1

        print(f"[QualityDatasetV3] 生成样本: {loaded_samples}, 跳过: {skipped_tasks}")

    def _build_sample(
        self,
        task_data: Dict[str, Any],
        current_image: str,
        other_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        label: int,
        subtask_idx: int,
        model: str,
        task_id: str
    ) -> Dict[str, Any]:
        """构建单个训练样本"""
        sample = build_quality_eval_sample_v3(
            task_data=task_data,
            current_result_image=current_image,
            other_result_images=other_images,
            input_images=input_images,
            ref_images=ref_images,
            label=label,
            subtask_idx=subtask_idx,
            shuffle_context=self.shuffle_context
        )

        # 跟踪临时文件
        if 'temp_files' in sample:
            self._temp_files.update(sample['temp_files'])

        # 添加元信息
        sample['model'] = model
        sample['task_id'] = task_id
        sample['subtask_idx'] = subtask_idx

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]

        if self.processor:
            try:
                messages = sample['messages']

                processed = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False
                )

                inputs = {}
                for k, v in processed.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() > 1 and v.size(0) == 1:
                            inputs[k] = v.squeeze(0).contiguous()
                        else:
                            inputs[k] = v.contiguous()
                    else:
                        inputs[k] = v

                inputs['labels'] = torch.tensor(sample['label'], dtype=torch.long)

                return inputs
            except Exception as e:
                print(f"\n[QualityDatasetV3 ERROR] 处理样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to process sample {idx}: {e}")

        return sample


@dataclass
class QualityEvalCollatorV3:
    """
    图像质量评估任务的自定义 Collator（V3版本）

    与V2的唯一区别：将标签字段重命名为 concept_labels
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """批处理多个样本"""
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        labels_list = []

        for feature in features:
            # 文本部分
            if 'input_ids' in feature:
                ids = feature['input_ids']
                if isinstance(ids, torch.Tensor):
                    input_ids_list.append(ids)
                else:
                    input_ids_list.append(torch.tensor(ids, dtype=torch.long))

            if 'attention_mask' in feature:
                mask = feature['attention_mask']
                if isinstance(mask, torch.Tensor):
                    attention_mask_list.append(mask)
                else:
                    attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

            # 图像部分
            if 'pixel_values' in feature:
                pv = feature['pixel_values']
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pixel_values_list.append(pv)

            if 'image_grid_thw' in feature:
                thw = feature['image_grid_thw']
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                image_grid_thw_list.append(thw)

            # 标签部分
            if 'labels' in feature:
                label = feature['labels']
                if isinstance(label, torch.Tensor):
                    if label.dim() == 0:
                        labels_list.append(label)
                    else:
                        labels_list.append(label.squeeze())
                else:
                    labels_list.append(torch.tensor(label, dtype=torch.long))

        # 构建批次字典
        batch = {}

        # Pad input_ids 和 attention_mask
        if input_ids_list:
            max_length = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(ids)

                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_ids)

                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)

            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_mask)

        # Stack pixel_values
        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        # Stack image_grid_thw
        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        # Stack labels - 改名为 concept_labels
        if labels_list:
            batch['concept_labels'] = torch.stack(labels_list)

        return batch


# ============================================================================
# 第四部分：V4 - 多维度评分（已过时，仅保留兼容性）
# ============================================================================
# 说明：7维度详细评分任务，推荐使用第五部分的统一架构（Stage1）
# ============================================================================

def build_system_prompt_v4() -> str:
    """
    构建系统提示词（V4版本）

    特点：提供完整的7维度评估指南，用于详细评分任务

    维度：
    1. BRF (Baseline Requirements Fulfillment) - 0-5分
    2. VEQ-Clarity (Clarity & Detail) - 0-5分
    3. VEQ-Realism (Realism & Artifacts) - 0-5分
    4. VEQ-Aesthetic (Aesthetic Quality) - 0-5分
    5. VEQ-Text (Text Quality) - 0-5分或N/A
    6. CNS-Edit (Edit Consistency) - 1-5分或N/A
    7. CNS-Set (Set Consistency) - 1-5分或N/A
    """
    prompt = """You are a professional image quality evaluator. Your task is to evaluate generated images across three major dimensions and predict scores for seven sub-metrics.

# DIMENSION 1: BRF (Baseline Requirements Fulfillment)

**Purpose**: Assess whether the generated image meets the task requirements.

**Evaluation Method**:
You will be given a list of evaluation points (requirements checklist). For each point:
- Score 0: The requirement is NOT met
- Score 1: The requirement IS met
- Do NOT use N/A - you must choose 0 or 1 for every point

**Scoring Rules**:
1. Check each evaluation point strictly against the image
2. Only give 1 if the requirement is clearly and fully met
3. If you cannot determine from the image alone, give 0
4. If the requirement is unclear or not applicable, give 0

**Final BRF Score Calculation** (0-5 scale):
BRF Score = (Number of completed points / Total points) × 5

**Important Notes**:
- Evaluate ONLY based on what is visible in the image
- Do NOT make assumptions about unseen aspects
- Do NOT adjust score based on visual quality (that's VEQ's job)
- Focus purely on requirement compliance

# DIMENSION 2: VEQ (Visual Execution Quality)

## 2.1 Technical Quality - Clarity & Detail (0-5)

**Focus**: Evaluate the perceptual sharpness and detail richness.

**Scoring Scale**:
- 5: Exceptional clarity - Exceptionally sharp with crisp edges, extremely rich in detail
- 4: Good clarity - Good sharpness with clear details, minor softness in limited areas
- 3: Moderate clarity - Moderate sharpness, some details lost or blurred
- 2: Poor clarity - Poor sharpness with significant blur, many details unclear
- 1: Very poor clarity - Severe blur, very few recognizable details
- 0: Unacceptable - Completely corrupted or unrecognizable

## 2.2 Technical Quality - Realism & Artifacts (0-5)

**Focus**: Evaluate whether the image appears natural and photorealistic.

**Scoring Scale**:
- 5: Completely photorealistic - No detectable AI artifacts, perfectly natural
- 4: Mostly realistic - Minor inconsistencies that don't affect overall realism
- 3: Moderately realistic - Noticeable AI artifacts but generally acceptable
- 2: Clear synthetic appearance - Obvious AI artifacts, implausible physics
- 1: Heavily synthetic - Pervasive artifacts, highly unnatural
- 0: Completely unrealistic - Obviously AI-generated with severe artifacts

## 2.3 Aesthetic Quality (0-5)

**Focus**: Assess the artistic and aesthetic value.

**Scoring Scale**:
- 5: Exceptional - Masterful composition, stunning colors, expert lighting
- 4: Strong - Well-composed, pleasing colors, effective lighting
- 3: Adequate - Acceptable composition, reasonable colors, functional lighting
- 2: Poor - Problematic composition, uncoordinated colors, inadequate lighting
- 1: Very poor - Chaotic composition, severe color conflicts
- 0: No aesthetic value - Completely unappealing

## 2.4 Text Quality (0-5 or N/A)

**Focus**: Assess the quality of text rendered in the image.

**Scoring Scale**:
- 5: Excellent - No errors, strong contrast, clear rendering, professional
- 4: Good - Minor or no errors, good contrast and readability
- 3: Adequate - 1-2 errors, weak contrast, average readability
- 2: Poor - Multiple errors, very weak contrast, difficult to read
- 1: Very poor - Largely unreadable, severe issues
- 0: Unreadable - Complete text failure
- N/A: No text present in the image


# DIMENSION 3: CNS (Consistency)

## 3.1 Edit Consistency - CNS-Edit (1-5 or N/A)

**Applicability**: ONLY for image editing tasks with source image.

**Scoring Scale**:
- 5: Perfect - No changes in unedited regions, seamless edges
- 4: Excellent - Barely perceptible artifacts, mostly coherent
- 3: Moderate - Localized contamination, clear seams
- 2: Poor - Multiple damaged areas, obvious mismatches
- 1: Failed - Large-scale unintended edits, identity lost
- N/A: Not an editing task (no source image provided)

## 3.2 Set Consistency - CNS-Set (1-5 or N/A)

**Applicability**: ONLY for multi-image tasks.

**Scoring Scale**:
- 5: Highly consistent - Style, colors, layout all uniform
- 4: Mostly consistent - 1 minor deviation
- 3: Moderately consistent - 2-3 moderate deviations
- 2: Poorly consistent - Multiple severe inconsistencies
- 1: Vastly inconsistent - Completely different from others
- N/A: Single image task

# OUTPUT FORMAT

Predict scores for all 7 dimensions:
1. BRF: 0-5
2. VEQ-Clarity: 0-5
3. VEQ-Realism: 0-5
4. VEQ-Aesthetic: 0-5
5. VEQ-Text: 0-5 or N/A
6. CNS-Edit: 1-5 or N/A
7. CNS-Set: 1-5 or N/A

Use N/A only when explicitly specified."""

    return prompt


def build_qwen3vl_message_v4(
    role_prompt: str,
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    subtask_idx: int,
) -> Dict[str, Any]:
    """
    构建Qwen3-VL的消息格式（V4版本）

    特点：
    - 用于7维度详细评分任务
    - 明确标注当前评估的图片
    - 提供完整的评估点清单（evaluation points）
    - 不使用随机打乱（确定性数据构造）

    Args:
        role_prompt: 角色提示词
        task_data: 任务JSON数据
        current_result_image: 当前要评估的结果图片路径
        other_result_images: 其他结果图片路径列表
        input_images: 任务输入图片路径列表
        ref_images: 任务参考图片路径列表
        subtask_idx: 当前评估的子任务索引（0-based）

    Returns:
        包含消息列表和图片列表的字典
    """
    # 提取任务信息
    original_data = task_data.get("original_task_data", {})
    requirements = original_data.get("requirements_original", "")
    output_quantity = original_data.get("output_quantity", 0)
    price_usd = original_data.get("price_usd", 0)

    # 提取子任务
    deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
    subtasks = []
    for i, d in enumerate(deliverables, 1):
        subtask_text = d.get("subtask", "")
        if subtask_text:
            subtasks.append(f"subtask_{i}: {subtask_text}")

    # 构建文本内容
    text_parts = []

    # 1. 任务需求
    text_parts.append(f"task_requirement: {requirements}")

    # 2. 子任务列表
    if subtasks:
        text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
        text_parts.append("\n".join(subtasks))

    # 3. 数字信息
    text_parts.append(f"\n\noutput_quantity: {output_quantity}")
    text_parts.append(f"task_value: ${price_usd}")

    # 4. 当前评估的子任务 - 明确指出评估点
    text_parts.append(f"\n\n## 当前评估任务")
    if subtask_idx < len(deliverables):
        current_deliverable = deliverables[subtask_idx]
        subtask_name = f"subtask_{subtask_idx + 1}"
        subtask_desc = current_deliverable.get("subtask", "")

        text_parts.append(f"\nEvaluating: {subtask_name}")
        if subtask_desc:
            text_parts.append(f'Requirement: "{subtask_desc}"')

        # 提取评估点（evaluation points）
        evaluation_points = current_deliverable.get("evaluation_points", [])
        if evaluation_points:
            text_parts.append(f"\n### Evaluation Points Checklist ({len(evaluation_points)} points):")
            for ep_idx, ep in enumerate(evaluation_points, 1):
                text_parts.append(f"{ep_idx}. {ep}")
        else:
            text_parts.append(f"\n### Evaluation Points: Check if the image meets the requirement stated above.")
    else:
        subtask_name = f"subtask_{subtask_idx + 1}"
        text_parts.append(f"\nEvaluating: {subtask_name}")

    # 5. 图片部分 - 顺序固定，不随机打乱
    text_parts.append(f"\n\n## 图片")

    # 当前评估图片（第一个，最重要）
    text_parts.append(f"\n### 当前评估图片 (Image to Evaluate)")
    text_parts.append(f"current_result_image: <image>")
    text_parts.append(
        f"Note: This is the PRIMARY image you need to evaluate across all 7 dimensions."
    )

    # 输入图片（如果有）
    if input_images:
        text_parts.append(f"\n### 任务输入图片 ({len(input_images)}张)")
        text_parts.append(f"Note: Original input images (for edit consistency evaluation).")
        for i in range(len(input_images)):
            text_parts.append(f"input_image_{i+1}: <image>")

    # 其他结果图片（固定顺序，不打乱）
    if other_result_images:
        text_parts.append(
            f"\n### 其他结果图片 (Other Generated Images, {len(other_result_images)}张)"
        )
        text_parts.append(f"Note: Other generated images for set consistency evaluation.")
        for i in range(len(other_result_images)):
            text_parts.append(f"other_result_{i+1}: <image>")

    # 参考图片
    if ref_images:
        text_parts.append(f"\n### 任务参考图片 ({len(ref_images)}张)")
        text_parts.append(f"Note: Reference images provided in the task description.")
        for i in range(len(ref_images)):
            text_parts.append(f"ref_image_{i+1}: <image>")

    user_text = "\n".join(text_parts)

    # 构建消息列表
    messages = []

    if role_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": role_prompt}]
        })

    # 用户消息 - 图片顺序固定：当前图片 → 输入图片 → 其他结果图片 → 参考图片
    user_content = []
    all_images = [current_result_image]
    all_images.extend(input_images)
    all_images.extend(other_result_images)
    all_images.extend(ref_images)

    user_content.append({"type": "text", "text": user_text})

    for img_path in all_images:
        user_content.append({"type": "image", "image": img_path})

    messages.append({
        "role": "user",
        "content": user_content
    })

    return {
        "messages": messages,
        "images": all_images,
        "metadata": {
            "task_id": task_data.get("task_id", ""),
            "subtask_idx": subtask_idx,
            "num_input_images": len(input_images),
            "num_ref_images": len(ref_images),
            "num_other_results": len(other_result_images),
            "total_images": len(all_images)
        }
    }


def build_quality_eval_sample_v4(
    task_data: Dict[str, Any],
    current_result_image: str,
    other_result_images: List[str],
    input_images: List[str],
    ref_images: List[str],
    scores: Dict[str, Optional[float]],
    subtask_idx: int,
) -> Dict[str, Any]:
    """
    构建单个训练样本（V4版本）

    Args:
        task_data: 任务数据字典
        current_result_image: 当前评估的图片路径
        other_result_images: 其他结果图片路径列表
        input_images: 输入图片路径列表
        ref_images: 参考图片路径列表
        scores: 7维度评分字典，格式：
            {
                "brf": float (0-5),
                "veq_clarity": float (0-5),
                "veq_realism": float (0-5),
                "veq_aesthetic": float (0-5),
                "veq_text": float or None (0-5 or N/A),
                "cns_edit": float or None (1-5 or N/A),
                "cns_set": float or None (1-5 or N/A)
            }
        subtask_idx: 当前评估的子任务索引

    Returns:
        {
            "messages": [...],
            "images": [...],
            "scores": dict,
            "task_id": str
        }
    """
    system_prompt = build_system_prompt_v4()

    message_data = build_qwen3vl_message_v4(
        role_prompt=system_prompt,
        task_data=task_data,
        current_result_image=current_result_image,
        other_result_images=other_result_images,
        input_images=input_images,
        ref_images=ref_images,
        subtask_idx=subtask_idx
    )

    return {
        **message_data,
        "scores": scores,
        "task_id": task_data.get("task_id", "")
    }


class QualityEvalDatasetV4(Dataset):
    """
    图像质量评估数据集（V4版本）

    特点：
    - 用于7维度详细评分任务
    - 每个子任务生成一个样本
    - 标签是7个维度的评分字典
    - 不使用随机打乱（确定性数据构造）
    """
    def __init__(
        self,
        train_jsonl_path: str,
        bench_dir: str,
        dataset_dir: str,
        processor=None
    ):
        """
        Args:
            train_jsonl_path: 训练数据 jsonl 文件路径
            bench_dir: ServImage_Bench 目录路径
            dataset_dir: ServImage_Dataset 目录路径
            processor: Qwen3VL 的 processor
        """
        self.processor = processor
        self.bench_dir = Path(bench_dir)
        self.dataset_dir = Path(dataset_dir)

        # 加载任务定义
        print(f"[QualityDatasetV4] 加载任务定义: {bench_dir}")
        self.bench_tasks = load_bench_tasks(bench_dir)

        # 加载训练数据
        print(f"[QualityDatasetV4] 加载训练数据: {train_jsonl_path}")
        self.samples = []
        self._load_and_expand_samples(train_jsonl_path)

        print(f"[QualityDatasetV4] 总共加载 {len(self.samples)} 个训练样本（7维评分）")

    def _load_and_expand_samples(self, jsonl_path: str):
        """
        从 jsonl 文件加载数据并展开为子任务样本

        期望的 JSONL 格式：
        {
            "model": "flux",
            "task_id": "portrait-001",
            "category": "C-Portrait",
            "scores": [
                {
                    "subtask_idx": 0,
                    "brf": 4.5,
                    "veq_clarity": 4.0,
                    "veq_realism": 3.5,
                    "veq_aesthetic": 4.2,
                    "veq_text": null,
                    "cns_edit": null,
                    "cns_set": 4.0
                },
                ...
            ]
        }
        """
        loaded_tasks = 0
        loaded_samples = 0
        skipped_tasks = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    model = data['model']
                    task_id = data['task_id']
                    category = data['category']
                    scores_list = data['scores']

                    if task_id not in self.bench_tasks:
                        print(f"[QualityDatasetV4] 警告: 任务 {task_id} 未在 Bench 中找到，跳过")
                        skipped_tasks += 1
                        continue

                    task_data = self.bench_tasks[task_id]

                    folder_name = task_id_to_folder_name(task_id, category)
                    result_images = get_images_from_folder(
                        self.dataset_dir / model / folder_name
                    )
                    task_num = int(task_id.split('-')[-1])
                    input_images = get_images_from_folder(
                        self.bench_dir / category / "inputs" / f"task_{task_num}"
                    )
                    ref_images = get_images_from_folder(
                        self.bench_dir / category / "refs" / f"task_{task_num}"
                    )

                    if not result_images:
                        skipped_tasks += 1
                        continue

                    for score_item in scores_list:
                        subtask_idx = score_item['subtask_idx']

                        if subtask_idx < len(result_images):
                            current_image = result_images[subtask_idx]
                            other_images = (
                                result_images[:subtask_idx] +
                                result_images[subtask_idx+1:]
                            )

                            scores = {
                                "brf": score_item.get("brf"),
                                "veq_clarity": score_item.get("veq_clarity"),
                                "veq_realism": score_item.get("veq_realism"),
                                "veq_aesthetic": score_item.get("veq_aesthetic"),
                                "veq_text": score_item.get("veq_text"),
                                "cns_edit": score_item.get("cns_edit"),
                                "cns_set": score_item.get("cns_set"),
                            }

                            sample = self._build_sample(
                                task_data=task_data,
                                current_image=current_image,
                                other_images=other_images,
                                input_images=input_images,
                                ref_images=ref_images,
                                scores=scores,
                                model=model,
                                task_id=task_id,
                                subtask_idx=subtask_idx
                            )

                            self.samples.append(sample)
                            loaded_samples += 1

                    loaded_tasks += 1

                except Exception as e:
                    print(f"[QualityDatasetV4] 错误: 第 {line_num} 行处理失败 - {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_tasks += 1

        print(f"[QualityDatasetV4] 处理任务: {loaded_tasks}, 生成样本: {loaded_samples}, 跳过: {skipped_tasks}")

    def _build_sample(
        self,
        task_data: Dict[str, Any],
        current_image: str,
        other_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        scores: Dict[str, Optional[float]],
        model: str,
        task_id: str,
        subtask_idx: int
    ) -> Dict[str, Any]:
        """构建单个训练样本"""
        sample = build_quality_eval_sample_v4(
            task_data=task_data,
            current_result_image=current_image,
            other_result_images=other_images,
            input_images=input_images,
            ref_images=ref_images,
            scores=scores,
            subtask_idx=subtask_idx
        )

        sample['model'] = model
        sample['task_id'] = task_id
        sample['subtask_idx'] = subtask_idx

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]

        if self.processor:
            try:
                messages = sample['messages']

                processed = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False
                )

                inputs = {}
                for k, v in processed.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() > 1 and v.size(0) == 1:
                            inputs[k] = v.squeeze(0).contiguous()
                        else:
                            inputs[k] = v.contiguous()
                    else:
                        inputs[k] = v

                inputs['scores'] = sample['scores']

                return inputs
            except Exception as e:
                print(f"\n[QualityDatasetV4 ERROR] 处理样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to process sample {idx}: {e}")

        return sample


@dataclass
class QualityEvalCollatorV4:
    """
    图像质量评估任务的自定义 Collator（V4版本）

    特点：
    - scores 是7个维度的评分字典
    - 处理 None 值（N/A）用-1表示
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """批处理多个样本"""
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        scores_dict_list = []

        for feature in features:
            if 'input_ids' in feature:
                ids = feature['input_ids']
                if isinstance(ids, torch.Tensor):
                    input_ids_list.append(ids)
                else:
                    input_ids_list.append(torch.tensor(ids, dtype=torch.long))

            if 'attention_mask' in feature:
                mask = feature['attention_mask']
                if isinstance(mask, torch.Tensor):
                    attention_mask_list.append(mask)
                else:
                    attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

            if 'pixel_values' in feature:
                pv = feature['pixel_values']
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pixel_values_list.append(pv)

            if 'image_grid_thw' in feature:
                thw = feature['image_grid_thw']
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                image_grid_thw_list.append(thw)

            if 'scores' in feature:
                scores_dict_list.append(feature['scores'])

        batch = {}

        if input_ids_list:
            max_length = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(ids)

                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_ids)

                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)

            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_mask)

        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        if scores_dict_list:
            def score_to_tensor(score):
                if score is None:
                    return -1.0
                return float(score)

            batch['scores'] = {
                'brf': torch.tensor([score_to_tensor(s['brf']) for s in scores_dict_list], dtype=torch.float),
                'veq_clarity': torch.tensor([score_to_tensor(s['veq_clarity']) for s in scores_dict_list], dtype=torch.float),
                'veq_realism': torch.tensor([score_to_tensor(s['veq_realism']) for s in scores_dict_list], dtype=torch.float),
                'veq_aesthetic': torch.tensor([score_to_tensor(s['veq_aesthetic']) for s in scores_dict_list], dtype=torch.float),
                'veq_text': torch.tensor([score_to_tensor(s['veq_text']) for s in scores_dict_list], dtype=torch.float),
                'cns_edit': torch.tensor([score_to_tensor(s['cns_edit']) for s in scores_dict_list], dtype=torch.float),
                'cns_set': torch.tensor([score_to_tensor(s['cns_set']) for s in scores_dict_list], dtype=torch.float),
            }

        return batch


# ============================================================================
# 第五部分：统一架构 - 两阶段训练管道（Base/Stage1/Stage2）
# ============================================================================
"""
统一架构设计：
- 所有任务共享相同的数据加载和图片压缩逻辑
- 只在任务prompt和输出格式上有区别

Base: Accept/Reject 预测（二分类，0/1标签）- 独立基础模型
Stage1: 7维度详细评分（BRF, VEQ-Clarity, VEQ-Realism, VEQ-Aesthetic, VEQ-Text, CNS-Edit, CNS-Set）- 第一阶段训练
Stage2: 基于Stage1特征的Accept/Reject判断（7维度评分 + 综合判断）- 第二阶段训练

训练策略：
- Base: 独立训练的二分类模型
- Stage1: 训练VLM + LoRA学习7维度评分，学习图像质量理解
- Stage2: 冻结Stage1模型，提取特征层，与原始VLM特征拼接，训练新的二分类头
"""


# 5.1 统一基类（包含共享的图片压缩数据加载逻辑）
class QualityEvalDatasetUnified(Dataset):
    """
    统一的图像质量评估数据集基类

    特点：
    - 所有任务共享相同的图片压缩数据加载逻辑
    - 子类只需实现任务特定的prompt构造和标签处理
    """

    def __init__(
        self,
        train_jsonl_path: str,
        bench_dir: str,
        dataset_dir: str,
        task_mode: str,  # "base", "stage1", "stage2"
        processor=None
    ):
        """
        Args:
            train_jsonl_path: 训练数据 jsonl 文件路径
            bench_dir: ServImage_Bench 目录路径
            dataset_dir: ServImage_Dataset 目录路径
            task_mode: 任务模式 ("base", "stage1", "stage2")
            processor: Qwen3VL 的 processor
        """
        self.task_mode = task_mode
        self.processor = processor
        self.bench_dir = Path(bench_dir)
        self.dataset_dir = Path(dataset_dir)

        # 临时文件管理（用于图片压缩）
        self._temp_files = set()
        atexit.register(self._cleanup_temp_files)

        # 加载任务定义
        print(f"[QualityDatasetUnified-{task_mode}] 加载任务定义: {bench_dir}")
        self.bench_tasks = load_bench_tasks(bench_dir)

        # 加载训练数据
        print(f"[QualityDatasetUnified-{task_mode}] 加载训练数据: {train_jsonl_path}")
        self.samples = []
        self._load_and_expand_samples(train_jsonl_path)

        print(f"[QualityDatasetUnified-{task_mode}] 总共加载 {len(self.samples)} 个训练样本")

    def _cleanup_temp_files(self):
        """清理所有生成的临时文件"""
        if not self._temp_files:
            return

        print(f"[QualityDatasetUnified] 清理 {len(self._temp_files)} 个临时文件...")
        cleaned = 0
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    cleaned += 1
            except Exception as e:
                print(f"[QualityDatasetUnified] 警告: 无法删除临时文件 {temp_file}: {e}")

        print(f"[QualityDatasetUnified] 已清理 {cleaned}/{len(self._temp_files)} 个临时文件")
        self._temp_files.clear()

    def __del__(self):
        """析构函数：确保对象销毁时清理临时文件"""
        self._cleanup_temp_files()

    def _load_images_with_compression(
        self,
        result_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        current_idx: int
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        统一的图片加载和压缩逻辑（所有任务共用）

        策略：
        - 当前结果图：1张，原样输入
        - 输入图：最多1张，原样输入
        - 其它结果图：压缩成最多2张拼贴图
        - 参考图：压缩成最多2张拼贴图

        改进：
        - 支持缓存压缩结果到 data/train/compressed_images
        - 缓存文件不会被清理，可供检查和复用

        Args:
            result_images: 所有结果图片路径列表
            input_images: 输入图片路径列表
            ref_images: 参考图片路径列表
            current_idx: 当前要评估的图片索引

        Returns:
            (current_image, input_images_to_use, other_result_composites, ref_composites)
        """
        # 1. 当前评估图片
        current_image = result_images[current_idx]

        # 2. 输入图片：最多1张
        input_images_to_use = input_images[:1] if input_images else []

        # 3. 其它结果图片
        other_result_images = (
            result_images[:current_idx] +
            result_images[current_idx+1:]
        )

        # 压缩其它结果图片成最多2张拼贴图（使用缓存）
        other_result_composites = compress_images_to_composites(
            other_result_images,
            max_composites=2,
            canvas_size=(224, 224),
            use_cache=True  # 启用缓存
        )
        # 注意：缓存文件不添加到 _temp_files，因为它们应该被保留

        # 4. 参考图片：压缩成最多2张拼贴图（使用缓存）
        ref_composites = compress_images_to_composites(
            ref_images,
            max_composites=2,
            canvas_size=(224, 224),
            use_cache=True  # 启用缓存
        )
        # 注意：缓存文件不添加到 _temp_files，因为它们应该被保留

        return current_image, input_images_to_use, other_result_composites, ref_composites

    def _load_and_expand_samples(self, jsonl_path: str):
        """
        从 jsonl 文件加载数据并展开为子任务样本

        支持的数据格式根据 task_mode 而定（单行单个子任务格式）：
        - base: {"model": ..., "task_id": ..., "category": ..., "subtask_idx": 0, "label": 1}
        - stage1: {"model": ..., "task_id": ..., "category": ..., "subtask_idx": 0, "brf_overall_score": 2.0, "veq_clarity": 5, ...}
        - stage2: {"model": ..., "task_id": ..., "category": ..., "subtask_idx": 0, "brf_overall_score": 2.0, ..., "label": 1}
        """
        loaded_samples = 0
        skipped_samples = 0

        # 记录跳过的样本信息（用于后续保存）
        skipped_records = []
        loaded_records = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    model = data['model']
                    task_id = data['task_id']
                    category = data['category']
                    subtask_idx = data.get('subtask_idx', 0)

                    if task_id not in self.bench_tasks:
                        print(f"[QualityDatasetUnified-{self.task_mode}] 警告: 任务 {task_id} 未在 Bench 中找到，跳过 (行 {line_num})")
                        skipped_samples += 1
                        skipped_records.append({
                            "line_num": line_num,
                            "reason": "task_not_in_bench",
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "original_data": data
                        })
                        continue

                    task_data = self.bench_tasks[task_id]

                    # 构建图片路径
                    folder_name = task_id_to_folder_name(task_id, category)
                    result_images = get_images_from_folder(
                        self.dataset_dir / model / folder_name
                    )

                    # 处理task_id格式 (task_353 或 product-263)
                    if 'task_' in task_id:
                        task_num = int(task_id.split('_')[-1])
                    else:
                        task_num = int(task_id.split('-')[-1])

                    input_images = get_images_from_folder(
                        self.bench_dir / category / "inputs" / f"task_{task_num}"
                    )
                    ref_images = get_images_from_folder(
                        self.bench_dir / category / "refs" / f"task_{task_num}"
                    )

                    if not result_images:
                        print(f"[QualityDatasetUnified-{self.task_mode}] 警告: 任务 {task_id} 没有结果图片，跳过 (行 {line_num})")
                        skipped_samples += 1
                        skipped_records.append({
                            "line_num": line_num,
                            "reason": "no_result_images",
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "original_data": data
                        })
                        continue

                    if subtask_idx >= len(result_images):
                        print(f"[QualityDatasetUnified-{self.task_mode}] 警告: 任务 {task_id} 子任务索引 {subtask_idx} 超出范围（最大 {len(result_images)-1}），跳过 (行 {line_num})")
                        skipped_samples += 1
                        skipped_records.append({
                            "line_num": line_num,
                            "reason": "subtask_index_out_of_range",
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "max_subtask_idx": len(result_images) - 1,
                            "original_data": data
                        })
                        continue

                    # 根据 task_mode 处理不同的数据格式
                    if self.task_mode == "base":
                        # Base: 单行单个子任务，包含 label 字段
                        label = data.get('label')
                        if label is None:
                            print(f"[QualityDatasetUnified-base] 警告: 缺少 label 字段，跳过 (行 {line_num})")
                            skipped_samples += 1
                            skipped_records.append({
                                "line_num": line_num,
                                "reason": "missing_label",
                                "task_id": task_id,
                                "model": model,
                                "category": category,
                                "subtask_idx": subtask_idx,
                                "original_data": data
                            })
                            continue

                        sample = self._build_sample_base(
                            task_data=task_data,
                            result_images=result_images,
                            input_images=input_images,
                            ref_images=ref_images,
                            subtask_idx=subtask_idx,
                            label=label,
                            model=model,
                            task_id=task_id
                        )
                        self.samples.append(sample)
                        loaded_samples += 1
                        loaded_records.append({
                            "line_num": line_num,
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "label": label,
                            "num_result_images": len(result_images),
                            "num_input_images": len(input_images),
                            "num_ref_images": len(ref_images)
                        })

                    elif self.task_mode == "stage1":
                        # Stage1: 单行单个子任务，包含7维度评分字段
                        # 字段名：brf_overall_score, veq_clarity, veq_realism, veq_aesthetic, veq_text,
                        #        cns_edit_consistency, cns_set_consistency
                        scores = {
                            "brf": data.get("brf_overall_score"),
                            "veq_clarity": data.get("veq_clarity"),
                            "veq_realism": data.get("veq_realism"),
                            "veq_aesthetic": data.get("veq_aesthetic"),
                            "veq_text": data.get("veq_text"),
                            "cns_edit": data.get("cns_edit_consistency"),
                            "cns_set": data.get("cns_set_consistency"),
                        }

                        # 检查必要字段
                        if scores["brf"] is None:
                            print(f"[QualityDatasetUnified-stage1] 警告: 缺少 brf_overall_score 字段，跳过 (行 {line_num})")
                            skipped_samples += 1
                            skipped_records.append({
                                "line_num": line_num,
                                "reason": "missing_brf_score",
                                "task_id": task_id,
                                "model": model,
                                "category": category,
                                "subtask_idx": subtask_idx,
                                "original_data": data
                            })
                            continue

                        sample = self._build_sample_stage1(
                            task_data=task_data,
                            result_images=result_images,
                            input_images=input_images,
                            ref_images=ref_images,
                            subtask_idx=subtask_idx,
                            scores=scores,
                            model=model,
                            task_id=task_id
                        )
                        self.samples.append(sample)
                        loaded_samples += 1
                        loaded_records.append({
                            "line_num": line_num,
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "scores": scores,
                            "num_result_images": len(result_images),
                            "num_input_images": len(input_images),
                            "num_ref_images": len(ref_images)
                        })

                    elif self.task_mode == "stage2":
                        # Stage2: 单行单个子任务，包含7维度评分 + label
                        scores = {
                            "brf": data.get("brf_overall_score"),
                            "veq_clarity": data.get("veq_clarity"),
                            "veq_realism": data.get("veq_realism"),
                            "veq_aesthetic": data.get("veq_aesthetic"),
                            "veq_text": data.get("veq_text"),
                            "cns_edit": data.get("cns_edit_consistency"),
                            "cns_set": data.get("cns_set_consistency"),
                        }

                        # Stage2 需要 label（如果没有，可以根据brf推断）
                        label = data.get("label")
                        if label is None:
                            # 根据 brf 推断：brf >= 3.0 认为接受，否则拒绝
                            brf = scores.get("brf")
                            if brf is not None:
                                label = 1 if brf >= 3.0 else 0
                                print(f"[QualityDatasetUnified-stage2] 警告: 缺少 label 字段，根据 brf={brf} 推断为 {label} (行 {line_num})")
                            else:
                                print(f"[QualityDatasetUnified-stage2] 错误: 既缺少 label 也缺少 brf_overall_score，跳过 (行 {line_num})")
                                skipped_samples += 1
                                skipped_records.append({
                                    "line_num": line_num,
                                    "reason": "missing_label_and_brf",
                                    "task_id": task_id,
                                    "model": model,
                                    "category": category,
                                    "subtask_idx": subtask_idx,
                                    "original_data": data
                                })
                                continue

                        sample = self._build_sample_stage2(
                            task_data=task_data,
                            result_images=result_images,
                            input_images=input_images,
                            ref_images=ref_images,
                            subtask_idx=subtask_idx,
                            scores=scores,
                            label=label,
                            model=model,
                            task_id=task_id
                        )
                        self.samples.append(sample)
                        loaded_samples += 1
                        loaded_records.append({
                            "line_num": line_num,
                            "task_id": task_id,
                            "model": model,
                            "category": category,
                            "subtask_idx": subtask_idx,
                            "label": label,
                            "scores": scores,
                            "num_result_images": len(result_images),
                            "num_input_images": len(input_images),
                            "num_ref_images": len(ref_images)
                        })

                except Exception as e:
                    print(f"[QualityDatasetUnified-{self.task_mode}] 错误: 第 {line_num} 行处理失败 - {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_samples += 1
                    skipped_records.append({
                        "line_num": line_num,
                        "reason": "exception",
                        "error_message": str(e),
                        "original_data": data if 'data' in locals() else None
                    })

        print(f"[QualityDatasetUnified-{self.task_mode}] 成功加载 {loaded_samples} 个样本，跳过 {skipped_samples} 个")

        # 保存整理后的数据集到指定目录
        self._save_dataset_records(jsonl_path, loaded_records, skipped_records)

    def _save_dataset_records(self, source_path: str, loaded_records: list, skipped_records: list):
        """
        保存数据集加载记录，方便手动清理

        Args:
            source_path: 原始数据文件路径
            loaded_records: 成功加载的样本记录
            skipped_records: 跳过的样本记录
        """
        import os
        from datetime import datetime

        # 确定保存目录
        save_dir = Path("data/train")
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确定数据集类型（从源文件路径推断）
        source_name = Path(source_path).stem  # train, val, test

        # 保存成功加载的样本
        loaded_file = save_dir / f"{self.task_mode}_{source_name}_loaded_{timestamp}.jsonl"
        with open(loaded_file, 'w', encoding='utf-8') as f:
            for record in loaded_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[QualityDatasetUnified-{self.task_mode}] 已保存成功加载的样本: {loaded_file}")

        # 保存跳过的样本
        if skipped_records:
            skipped_file = save_dir / f"{self.task_mode}_{source_name}_skipped_{timestamp}.jsonl"
            with open(skipped_file, 'w', encoding='utf-8') as f:
                for record in skipped_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"[QualityDatasetUnified-{self.task_mode}] 已保存跳过的样本: {skipped_file}")

            # 生成跳过样本的统计摘要
            summary_file = save_dir / f"{self.task_mode}_{source_name}_skipped_summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"数据集加载统计摘要\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"源文件: {source_path}\n")
                f.write(f"任务模式: {self.task_mode}\n")
                f.write(f"加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\n")
                f.write(f"成功加载: {len(loaded_records)} 个样本\n")
                f.write(f"跳过样本: {len(skipped_records)} 个样本\n")
                f.write(f"跳过比例: {len(skipped_records)/(len(loaded_records)+len(skipped_records))*100:.2f}%\n")
                f.write(f"\n")

                # 按原因统计跳过样本
                reason_counts = {}
                for record in skipped_records:
                    reason = record.get('reason', 'unknown')
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

                f.write(f"跳过原因统计:\n")
                f.write(f"-" * 60 + "\n")
                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {reason}: {count} 个样本\n")

                f.write(f"\n")
                f.write(f"详细信息请查看: {skipped_file}\n")

            print(f"[QualityDatasetUnified-{self.task_mode}] 已生成跳过样本统计摘要: {summary_file}")

    def _build_sample_base(
        self,
        task_data: Dict[str, Any],
        result_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        subtask_idx: int,
        label: int,
        model: str,
        task_id: str
    ) -> Dict[str, Any]:
        """构建 Base 样本（Accept/Reject 预测）"""
        # 使用统一的图片压缩逻辑
        current_image, input_imgs, other_composites, ref_composites = \
            self._load_images_with_compression(result_images, input_images, ref_images, subtask_idx)

        # 构建 Base 特定的 prompt
        system_prompt = self._build_system_prompt_base()
        messages, all_images = self._build_messages_base(
            system_prompt=system_prompt,
            task_data=task_data,
            current_image=current_image,
            input_images=input_imgs,
            other_composites=other_composites,
            ref_composites=ref_composites,
            subtask_idx=subtask_idx,
            num_other_originals=len(result_images) - 1,
            num_ref_originals=len(ref_images)
        )

        return {
            "messages": messages,
            "images": all_images,
            "label": label,
            "task_id": task_id,
            "model": model,
            "subtask_idx": subtask_idx
        }

    def _build_sample_stage1(
        self,
        task_data: Dict[str, Any],
        result_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        subtask_idx: int,
        scores: Dict[str, Optional[float]],
        model: str,
        task_id: str
    ) -> Dict[str, Any]:
        """构建 Stage1 样本（7维度详细评分）"""
        # 使用统一的图片压缩逻辑
        current_image, input_imgs, other_composites, ref_composites = \
            self._load_images_with_compression(result_images, input_images, ref_images, subtask_idx)

        # 构建 Stage1 特定的 prompt
        system_prompt = self._build_system_prompt_stage1()
        messages, all_images = self._build_messages_stage1(
            system_prompt=system_prompt,
            task_data=task_data,
            current_image=current_image,
            input_images=input_imgs,
            other_composites=other_composites,
            ref_composites=ref_composites,
            subtask_idx=subtask_idx,
            num_other_originals=len(result_images) - 1,
            num_ref_originals=len(ref_images)
        )

        return {
            "messages": messages,
            "images": all_images,
            "scores": scores,
            "task_id": task_id,
            "model": model,
            "subtask_idx": subtask_idx
        }

    def _build_sample_stage2(
        self,
        task_data: Dict[str, Any],
        result_images: List[str],
        input_images: List[str],
        ref_images: List[str],
        subtask_idx: int,
        scores: Dict[str, Optional[float]],
        label: int,
        model: str,
        task_id: str
    ) -> Dict[str, Any]:
        """构建 Stage2 样本（7维度评分 + Accept/Reject 综合判断）"""
        # 使用统一的图片压缩逻辑
        current_image, input_imgs, other_composites, ref_composites = \
            self._load_images_with_compression(result_images, input_images, ref_images, subtask_idx)

        # 构建 Stage2 特定的 prompt
        system_prompt = self._build_system_prompt_stage2()
        messages, all_images = self._build_messages_stage2(
            system_prompt=system_prompt,
            task_data=task_data,
            current_image=current_image,
            input_images=input_imgs,
            other_composites=other_composites,
            ref_composites=ref_composites,
            subtask_idx=subtask_idx,
            num_other_originals=len(result_images) - 1,
            num_ref_originals=len(ref_images)
        )

        return {
            "messages": messages,
            "images": all_images,
            "scores": scores,
            "label": label,
            "task_id": task_id,
            "model": model,
            "subtask_idx": subtask_idx
        }

    def _build_system_prompt_base(self) -> str:
        """构建 Base 的系统提示词（Accept/Reject 预测）"""
        prompt = """You are a professional image processing evaluation assistant.

I will provide you with:
- A task description with specific requirements and subtasks
- The CURRENT IMAGE that you need to evaluate (marked as "current_result_image")
- Other result images for context (may be shown as grid composites)
- Task input images (if this is an image modification task)
- Reference images (may be shown as grid composites)

Your task is to evaluate ONLY the current image to determine if it meets the specific requirement for the indicated subtask.

Note: Some images may be presented as grid composites containing multiple images for efficient visual token usage.
Please consider the task value when making your evaluation decision. Higher-value tasks may warrant more thorough evaluation of quality standards.
Output your evaluation as a binary decision: accept (1) or reject (0)."""

        return prompt

    def _build_messages_base(
        self,
        system_prompt: str,
        task_data: Dict[str, Any],
        current_image: str,
        input_images: List[str],
        other_composites: List[str],
        ref_composites: List[str],
        subtask_idx: int,
        num_other_originals: int,
        num_ref_originals: int
    ) -> Tuple[List[Dict], List[str]]:
        """构建 Base 的消息格式"""
        # 提取任务信息
        original_data = task_data.get("original_task_data", {})
        requirements = original_data.get("requirements_original", "")
        output_quantity = original_data.get("output_quantity", 0)
        price_usd = original_data.get("price_usd", 0)

        # 提取子任务
        deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
        subtasks = []
        for i, d in enumerate(deliverables, 1):
            subtask_text = d.get("subtask", "")
            if subtask_text:
                subtasks.append(f"subtask_{i}: {subtask_text}")

        # 构建文本内容
        text_parts = []
        text_parts.append(f"task_requirement: {requirements}")

        if subtasks:
            text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
            text_parts.append("\n".join(subtasks))

        text_parts.append(f"\n\noutput_quantity: {output_quantity}")
        text_parts.append(f"task_value: ${price_usd}")

        # 当前评估的子任务
        text_parts.append(f"\n\n## 图片")
        text_parts.append(f"\n### 当前评估图片 (Current Image to Evaluate)")

        if subtask_idx < len(subtasks):
            current_subtask = subtasks[subtask_idx]
            if ": " in current_subtask:
                subtask_name, subtask_desc = current_subtask.split(": ", 1)
            else:
                subtask_name = current_subtask
                subtask_desc = ""

            text_parts.append(f"Evaluating: {subtask_name}")
            if subtask_desc:
                text_parts.append(f'Requirement: "{subtask_desc}"')
        else:
            subtask_name = f"subtask_{subtask_idx + 1}"
            text_parts.append(f"Evaluating: {subtask_name}")

        text_parts.append(f"current_result_image: <image>")

        # 输入图片
        if input_images:
            text_parts.append(f"\n### 任务输入图片 ({len(input_images)}张)")
            for i in range(len(input_images)):
                text_parts.append(f"input_image_{i+1}: <image>")

        # 其它结果图片（压缩版）
        if other_composites:
            text_parts.append(
                f"\n### 其他结果图片 (Compressed from {num_other_originals} images into {len(other_composites)} composite(s))"
            )
            for i in range(len(other_composites)):
                text_parts.append(f"other_results_composite_{i+1}: <image>")

        # 参考图片（压缩版）
        if ref_composites:
            text_parts.append(
                f"\n### 任务参考图片 (Compressed from {num_ref_originals} images into {len(ref_composites)} composite(s))"
            )
            for i in range(len(ref_composites)):
                text_parts.append(f"ref_composite_{i+1}: <image>")

        user_text = "\n".join(text_parts)

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # 图片顺序：当前图片 → 输入图片 → 其它结果拼贴图 → 参考拼贴图
        all_images = [current_image]
        all_images.extend(input_images)
        all_images.extend(other_composites)
        all_images.extend(ref_composites)

        user_content = [{"type": "text", "text": user_text}]
        for img_path in all_images:
            user_content.append({"type": "image", "image": img_path})

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages, all_images

    # ========== Stage1: 7维度详细评分的 Prompt 构建 ==========
    def _build_system_prompt_stage1(self) -> str:
        """构建 Stage1 的系统提示词（7维度详细评分）"""
        # Stage1 使用与 V4 相同的7维度详细评分指南
        return build_system_prompt_v4()

    def _build_messages_stage1(
        self,
        system_prompt: str,
        task_data: Dict[str, Any],
        current_image: str,
        input_images: List[str],
        other_composites: List[str],
        ref_composites: List[str],
        subtask_idx: int,
        num_other_originals: int,
        num_ref_originals: int
    ) -> Tuple[List[Dict], List[str]]:
        """构建 Stage1 的消息格式（7维度详细评分）"""
        # 提取任务信息
        original_data = task_data.get("original_task_data", {})
        requirements = original_data.get("requirements_original", "")
        output_quantity = original_data.get("output_quantity", 0)
        price_usd = original_data.get("price_usd", 0)

        # 提取子任务
        deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
        subtasks = []
        for i, d in enumerate(deliverables, 1):
            subtask_text = d.get("subtask", "")
            if subtask_text:
                subtasks.append(f"subtask_{i}: {subtask_text}")

        # 构建文本内容
        text_parts = []
        text_parts.append(f"task_requirement: {requirements}")

        if subtasks:
            text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
            text_parts.append("\n".join(subtasks))

        text_parts.append(f"\n\noutput_quantity: {output_quantity}")
        text_parts.append(f"task_value: ${price_usd}")

        # 当前评估任务 - 添加 evaluation points
        text_parts.append(f"\n\n## 当前评估任务")
        if subtask_idx < len(deliverables):
            current_deliverable = deliverables[subtask_idx]
            subtask_name = f"subtask_{subtask_idx + 1}"
            subtask_desc = current_deliverable.get("subtask", "")

            text_parts.append(f"\nEvaluating: {subtask_name}")
            if subtask_desc:
                text_parts.append(f'Requirement: "{subtask_desc}"')

            # 添加评估点清单
            evaluation_points = current_deliverable.get("evaluation_points", [])
            if evaluation_points:
                text_parts.append(f"\n### Evaluation Points Checklist ({len(evaluation_points)} points):")
                for ep_idx, ep in enumerate(evaluation_points, 1):
                    text_parts.append(f"{ep_idx}. {ep}")
        else:
            subtask_name = f"subtask_{subtask_idx + 1}"
            text_parts.append(f"\nEvaluating: {subtask_name}")

        # 图片部分
        text_parts.append(f"\n\n## 图片")
        text_parts.append(f"\n### 当前评估图片 (Image to Evaluate)")
        text_parts.append(f"current_result_image: <image>")

        # 输入图片
        if input_images:
            text_parts.append(f"\n### 任务输入图片 ({len(input_images)}张)")
            for i in range(len(input_images)):
                text_parts.append(f"input_image_{i+1}: <image>")

        # 其它结果图片
        if other_composites:
            text_parts.append(
                f"\n### 其他结果图片 (Compressed from {num_other_originals} images into {len(other_composites)} composite(s))"
            )
            for i in range(len(other_composites)):
                text_parts.append(f"other_results_composite_{i+1}: <image>")

        # 参考图片
        if ref_composites:
            text_parts.append(
                f"\n### 任务参考图片 (Compressed from {num_ref_originals} images into {len(ref_composites)} composite(s))"
            )
            for i in range(len(ref_composites)):
                text_parts.append(f"ref_composite_{i+1}: <image>")

        user_text = "\n".join(text_parts)

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # 图片顺序
        all_images = [current_image]
        all_images.extend(input_images)
        all_images.extend(other_composites)
        all_images.extend(ref_composites)

        user_content = [{"type": "text", "text": user_text}]
        for img_path in all_images:
            user_content.append({"type": "image", "image": img_path})

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages, all_images

    # ========== Stage2: 7维度评分 + Accept/Reject 综合预测 ==========
    def _build_system_prompt_stage2(self) -> str:
        """构建 Stage2 的系统提示词（7维度评分 + Accept/Reject 综合判断）"""
        # Stage2 在 Stage1 基础上增加 Accept/Reject 判断
        base_prompt = build_system_prompt_v4()

        # 添加综合判断说明
        additional_instruction = """

# FINAL DECISION: Accept/Reject

After predicting all 7 dimension scores, you need to make a final binary decision:
- Accept (1): The image is acceptable and meets the overall quality standards
- Reject (0): The image should be rejected due to insufficient quality

**Decision Guidelines**:
- Consider all 7 dimensions comprehensively
- BRF is the most critical - low BRF (< 2.0) usually means reject
- Severe issues in any VEQ dimension (score = 0) should lead to reject
- N/A scores should not directly cause rejection
- Use your professional judgment to balance different aspects
- Please consider the price of the task here when making your decision, and then determine whether the result of the output image here is worth accepting.

**Output Format**:
Output your evaluation as a binary decision: accept (1) or reject (0)."""

        return base_prompt + additional_instruction

    def _build_messages_stage2(
        self,
        system_prompt: str,
        task_data: Dict[str, Any],
        current_image: str,
        input_images: List[str],
        other_composites: List[str],
        ref_composites: List[str],
        subtask_idx: int,
        num_other_originals: int,
        num_ref_originals: int
    ) -> Tuple[List[Dict], List[str]]:
        """构建 Stage2 的消息格式"""
        # 类似 Stage1，但添加 evaluation points
        # 提取任务信息
        original_data = task_data.get("original_task_data", {})
        requirements = original_data.get("requirements_original", "")
        output_quantity = original_data.get("output_quantity", 0)
        price_usd = original_data.get("price_usd", 0)

        # 提取子任务
        deliverables = task_data.get("hard_rules", {}).get("deliverables", [])
        subtasks = []
        for i, d in enumerate(deliverables, 1):
            subtask_text = d.get("subtask", "")
            if subtask_text:
                subtasks.append(f"subtask_{i}: {subtask_text}")

        # 构建文本内容
        text_parts = []
        text_parts.append(f"task_requirement: {requirements}")

        if subtasks:
            text_parts.append(f"\n\nnum_subtasks: {len(subtasks)}")
            text_parts.append("\n".join(subtasks))

        text_parts.append(f"\n\noutput_quantity: {output_quantity}")
        text_parts.append(f"task_value: ${price_usd}")

        # 当前评估任务
        text_parts.append(f"\n\n## 当前评估任务")
        if subtask_idx < len(deliverables):
            current_deliverable = deliverables[subtask_idx]
            subtask_name = f"subtask_{subtask_idx + 1}"
            subtask_desc = current_deliverable.get("subtask", "")

            text_parts.append(f"\nEvaluating: {subtask_name}")
            if subtask_desc:
                text_parts.append(f'Requirement: "{subtask_desc}"')

            # 提取评估点
            evaluation_points = current_deliverable.get("evaluation_points", [])
            if evaluation_points:
                text_parts.append(f"\n### Evaluation Points Checklist ({len(evaluation_points)} points):")
                for ep_idx, ep in enumerate(evaluation_points, 1):
                    text_parts.append(f"{ep_idx}. {ep}")
        else:
            subtask_name = f"subtask_{subtask_idx + 1}"
            text_parts.append(f"\nEvaluating: {subtask_name}")

        # 图片部分
        text_parts.append(f"\n\n## 图片")
        text_parts.append(f"\n### 当前评估图片 (Image to Evaluate)")
        text_parts.append(f"current_result_image: <image>")

        # 输入图片
        if input_images:
            text_parts.append(f"\n### 任务输入图片 ({len(input_images)}张)")
            for i in range(len(input_images)):
                text_parts.append(f"input_image_{i+1}: <image>")

        # 其它结果图片
        if other_composites:
            text_parts.append(
                f"\n### 其他结果图片 (Compressed from {num_other_originals} images into {len(other_composites)} composite(s))"
            )
            for i in range(len(other_composites)):
                text_parts.append(f"other_results_composite_{i+1}: <image>")

        # 参考图片
        if ref_composites:
            text_parts.append(
                f"\n### 任务参考图片 (Compressed from {num_ref_originals} images into {len(ref_composites)} composite(s))"
            )
            for i in range(len(ref_composites)):
                text_parts.append(f"ref_composite_{i+1}: <image>")

        user_text = "\n".join(text_parts)

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # 图片顺序
        all_images = [current_image]
        all_images.extend(input_images)
        all_images.extend(other_composites)
        all_images.extend(ref_composites)

        user_content = [{"type": "text", "text": user_text}]
        for img_path in all_images:
            user_content.append({"type": "image", "image": img_path})

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages, all_images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]

        if self.processor:
            try:
                messages = sample['messages']

                processed = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False
                )

                inputs = {}
                for k, v in processed.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() > 1 and v.size(0) == 1:
                            inputs[k] = v.squeeze(0).contiguous()
                        else:
                            inputs[k] = v.contiguous()
                    else:
                        inputs[k] = v

                # 根据 task_mode 添加对应的标签字段
                if self.task_mode == "base":
                    inputs['labels'] = torch.tensor(sample['label'], dtype=torch.long)
                elif self.task_mode == "stage1":
                    inputs['scores'] = sample['scores']  # Stage1: 7维度评分
                elif self.task_mode == "stage2":
                    inputs['scores'] = sample['scores']  # Stage2: 7维度评分
                    inputs['label'] = torch.tensor(sample['label'], dtype=torch.long)  # + Accept/Reject

                return inputs
            except Exception as e:
                print(f"\n[QualityDatasetUnified ERROR] 处理样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to process sample {idx}: {e}")

        return sample


# 5.2 对应的 Collator 类
@dataclass
class QualityEvalCollatorUnified:
    """
    统一的图像质量评估任务 Collator

    根据 task_mode 处理不同的标签格式：
    - base: labels (batch_size,) - 0/1 标签
    - stage1: scores dict with 7 dimensions
    - stage2: scores dict with 7 dimensions + label (batch_size,) - 0/1 标签
    """

    def __init__(self, processor, task_mode: str):
        self.processor = processor
        self.task_mode = task_mode
        self.pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """批处理多个样本"""
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        # 根据 task_mode 收集不同的标签
        labels_list = []
        scores_3d_list = []
        scores_list = []

        for feature in features:
            # 文本部分
            if 'input_ids' in feature:
                ids = feature['input_ids']
                if isinstance(ids, torch.Tensor):
                    input_ids_list.append(ids)
                else:
                    input_ids_list.append(torch.tensor(ids, dtype=torch.long))

            if 'attention_mask' in feature:
                mask = feature['attention_mask']
                if isinstance(mask, torch.Tensor):
                    attention_mask_list.append(mask)
                else:
                    attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

            # 图像部分
            if 'pixel_values' in feature:
                pv = feature['pixel_values']
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pixel_values_list.append(pv)

            if 'image_grid_thw' in feature:
                thw = feature['image_grid_thw']
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                image_grid_thw_list.append(thw)

            # 标签部分（根据 task_mode）
            if self.task_mode == "base":
                if 'labels' in feature:
                    label = feature['labels']
                    if isinstance(label, torch.Tensor):
                        if label.dim() == 0:
                            labels_list.append(label)
                        else:
                            labels_list.append(label.squeeze())
                    else:
                        labels_list.append(torch.tensor(label, dtype=torch.long))

            elif self.task_mode == "stage1":
                if 'scores' in feature:
                    scores_list.append(feature['scores'])

            elif self.task_mode == "stage2":
                if 'scores' in feature:
                    scores_list.append(feature['scores'])
                if 'label' in feature:
                    label = feature['label']
                    if isinstance(label, torch.Tensor):
                        if label.dim() == 0:
                            labels_list.append(label)
                        else:
                            labels_list.append(label.squeeze())
                    else:
                        labels_list.append(torch.tensor(label, dtype=torch.long))

        # 构建批次字典
        batch = {}

        # Pad input_ids 和 attention_mask
        if input_ids_list:
            max_length = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(ids)

                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_ids)

                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)

            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_mask)

        # Stack pixel_values
        if pixel_values_list:
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        # Stack image_grid_thw
        if image_grid_thw_list:
            batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        # 处理标签（根据 task_mode）
        if self.task_mode == "base":
            if labels_list:
                batch['labels'] = torch.stack(labels_list)

        elif self.task_mode == "stage1":
            # Stage1: 7维度评分
            if scores_list:
                def score_to_tensor(score):
                    if score is None or score == 'N/A':
                        return -1.0
                    return float(score)

                batch['scores'] = {
                    'brf': torch.tensor([score_to_tensor(s['brf']) for s in scores_list], dtype=torch.float),
                    'veq_clarity': torch.tensor([score_to_tensor(s['veq_clarity']) for s in scores_list], dtype=torch.float),
                    'veq_realism': torch.tensor([score_to_tensor(s['veq_realism']) for s in scores_list], dtype=torch.float),
                    'veq_aesthetic': torch.tensor([score_to_tensor(s['veq_aesthetic']) for s in scores_list], dtype=torch.float),
                    'veq_text': torch.tensor([score_to_tensor(s['veq_text']) for s in scores_list], dtype=torch.float),
                    'cns_edit': torch.tensor([score_to_tensor(s['cns_edit']) for s in scores_list], dtype=torch.float),
                    'cns_set': torch.tensor([score_to_tensor(s['cns_set']) for s in scores_list], dtype=torch.float),
                }

        elif self.task_mode == "stage2":
            # Stage2: 7维度评分 + Accept/Reject标签
            if scores_list:
                def score_to_tensor(score):
                    if score is None or score == 'N/A':
                        return -1.0
                    return float(score)

                batch['scores'] = {
                    'brf': torch.tensor([score_to_tensor(s['brf']) for s in scores_list], dtype=torch.float),
                    'veq_clarity': torch.tensor([score_to_tensor(s['veq_clarity']) for s in scores_list], dtype=torch.float),
                    'veq_realism': torch.tensor([score_to_tensor(s['veq_realism']) for s in scores_list], dtype=torch.float),
                    'veq_aesthetic': torch.tensor([score_to_tensor(s['veq_aesthetic']) for s in scores_list], dtype=torch.float),
                    'veq_text': torch.tensor([score_to_tensor(s['veq_text']) for s in scores_list], dtype=torch.float),
                    'cns_edit': torch.tensor([score_to_tensor(s['cns_edit']) for s in scores_list], dtype=torch.float),
                    'cns_set': torch.tensor([score_to_tensor(s['cns_set']) for s in scores_list], dtype=torch.float),
                }
            if labels_list:
                batch['labels'] = torch.stack(labels_list)

        return batch


# ============================================================================
# 第六部分：便捷创建函数
# ============================================================================

def create_quality_eval_dataloader(
    train_jsonl_path: str,
    bench_dir: str,
    dataset_dir: str,
    processor,
    version: str = "v3",
    task_mode: str = None,  # "base", "stage1", "stage2" - 优先使用统一架构
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    shuffle_context: bool = False,
):
    """
    创建质量评估数据加载器

    Args:
        train_jsonl_path: 训练数据路径
        bench_dir: Bench目录
        dataset_dir: Dataset目录
        processor: Qwen processor
        version: 版本选择 ("v2", "v3", "v4") - 旧架构（已弃用，建议使用 task_mode）
            - v2: 子任务级评估（二分类标签）
            - v3: 子任务级 + 图片压缩（二分类标签，推荐）
            - v4: 多维度评分（7个维度详细评分）
        task_mode: 任务模式 ("base", "stage1", "stage2") - **推荐使用新统一架构**
            - base: Accept/Reject 预测（二分类，使用图片压缩）- 独立基础模型
            - stage1: 7维度详细评分（BRF, VEQ-Clarity, VEQ-Realism, VEQ-Aesthetic, VEQ-Text, CNS-Edit, CNS-Set）- 第一阶段训练
            - stage2: 基于Stage1特征的Accept/Reject判断（7维度评分 + 综合判断）- 第二阶段训练
            - 所有任务模式共享相同的数据加载和图片压缩逻辑
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        shuffle_context: 是否打乱上下文（仅V2/V3使用）

    Returns:
        DataLoader

    示例:
        # 推荐：使用统一架构
        dataloader = create_quality_eval_dataloader(
            train_jsonl_path="/path/to/train.jsonl",
            bench_dir="/path/to/ServImage_Bench",
            dataset_dir="/path/to/ServImage_Dataset",
            processor=processor,
            task_mode="base",  # 或 "stage1", "stage2"
            batch_size=8
        )

        # 旧方式：使用 version 参数（仍然支持）
        dataloader = create_quality_eval_dataloader(
            train_jsonl_path="/path/to/train.jsonl",
            bench_dir="/path/to/ServImage_Bench",
            dataset_dir="/path/to/ServImage_Dataset",
            processor=processor,
            version="v3",
            batch_size=8
        )
    """
    # 优先使用 task_mode（新统一架构）
    if task_mode is not None:
        if task_mode not in ["base", "stage1", "stage2"]:
            raise ValueError(f"未知任务模式: {task_mode}，请选择 'base', 'stage1', 或 'stage2'")

        dataset = QualityEvalDatasetUnified(
            train_jsonl_path=train_jsonl_path,
            bench_dir=bench_dir,
            dataset_dir=dataset_dir,
            task_mode=task_mode,
            processor=processor
        )
        collator = QualityEvalCollatorUnified(processor, task_mode=task_mode)

    # 使用 version（旧架构）
    elif version == "v2":
        dataset = QualityEvalDatasetV2(
            train_jsonl_path=train_jsonl_path,
            bench_dir=bench_dir,
            dataset_dir=dataset_dir,
            shuffle_context=shuffle_context,
            processor=processor
        )
        collator = QualityEvalCollatorV2(processor)

    elif version == "v3":
        dataset = QualityEvalDatasetV3(
            train_jsonl_path=train_jsonl_path,
            bench_dir=bench_dir,
            dataset_dir=dataset_dir,
            shuffle_context=shuffle_context,
            processor=processor
        )
        collator = QualityEvalCollatorV3(processor)

    elif version == "v4":
        dataset = QualityEvalDatasetV4(
            train_jsonl_path=train_jsonl_path,
            bench_dir=bench_dir,
            dataset_dir=dataset_dir,
            processor=processor
        )
        collator = QualityEvalCollatorV4(processor)

    else:
        raise ValueError(f"未知版本: {version}，请选择 'v2', 'v3', 或 'v4'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )

    return dataloader


# ============================================================================
# 第七部分：使用示例
# ============================================================================

if __name__ == "__main__":
    """
    使用示例
    """
    print("=" * 80)
    print("ServImage Quality Evaluation Dataset - 使用示例")
    print("=" * 80)

    # ========== 推荐：使用统一架构（Base/Stage1/Stage2）==========
    print("\n" + "=" * 80)
    print("【推荐】统一架构 - 两阶段训练管道")
    print("=" * 80)

    # 示例1: Base - Accept/Reject 预测
    print("\n示例1: Base - Accept/Reject 预测（独立基础模型）")
    print("-" * 80)
    example_code_base = '''
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train_base.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    task_mode="base",  # Accept/Reject 预测
    batch_size=8
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)  # 已压缩
    # batch['labels']: (batch_size,)  # 0/1 标签
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

# JSONL 格式示例 (train_base.jsonl):
{
    "model": "flux",
    "task_id": "portrait-001",
    "category": "C-Portrait",
    "labels": [1, 0, 1, 1],  # 4个子任务的标签
    "num_valid_subtasks": 4
}
'''
    print(example_code_base)

    # 示例2: Stage1 - 7维度详细评分
    print("\n示例2: Stage1 - 7维度详细评分（第一阶段训练）")
    print("-" * 80)
    example_code_stage1 = '''
dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train_stage1.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    task_mode="stage1",  # 7维度详细评分
    batch_size=4
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)  # 已压缩
    # batch['scores']: dict with 7 dimensions
    #   - 'brf': (batch_size,) - 0-5
    #   - 'veq_clarity': (batch_size,) - 0-5
    #   - 'veq_realism': (batch_size,) - 0-5
    #   - 'veq_aesthetic': (batch_size,) - 0-5
    #   - 'veq_text': (batch_size,) - 0-5 or -1 for N/A
    #   - 'cns_edit': (batch_size,) - 1-5 or -1 for N/A
    #   - 'cns_set': (batch_size,) - 1-5 or -1 for N/A
    outputs = model(**batch)
    loss = compute_multidim_loss(outputs, batch['scores'])
    loss.backward()

# JSONL 格式示例 (train_stage1.jsonl):
{
    "model": "flux",
    "task_id": "portrait-001",
    "category": "C-Portrait",
    "scores": [
        {
            "subtask_idx": 0,
            "brf": 4.5,
            "veq_clarity": 4.0,
            "veq_realism": 3.5,
            "veq_aesthetic": 4.2,
            "veq_text": null,  # N/A
            "cns_edit": null,  # N/A
            "cns_set": 4.0
        }
    ]
}
'''
    print(example_code_stage1)

    # 示例3: Stage2 - 7维度评分 + Accept/Reject 综合判断
    print("\n示例3: Stage2 - 基于Stage1特征的Accept/Reject判断（第二阶段训练）")
    print("-" * 80)
    example_code_stage2 = '''
dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train_stage2.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    task_mode="stage2",  # 7维度评分 + Accept/Reject
    batch_size=4
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)  # 已压缩
    # batch['scores']: dict with 7 dimensions
    #   - 'brf': (batch_size,) - 0-5
    #   - 'veq_clarity': (batch_size,) - 0-5
    #   - 'veq_realism': (batch_size,) - 0-5
    #   - 'veq_aesthetic': (batch_size,) - 0-5
    #   - 'veq_text': (batch_size,) - 0-5 or -1 for N/A
    #   - 'cns_edit': (batch_size,) - 1-5 or -1 for N/A
    #   - 'cns_set': (batch_size,) - 1-5 or -1 for N/A
    # batch['labels']: (batch_size,) - 0/1 Accept/Reject标签
    outputs = model(**batch)
    loss_scores = compute_multidim_loss(outputs.scores, batch['scores'])
    loss_labels = compute_classification_loss(outputs.labels, batch['labels'])
    loss = loss_scores + loss_labels
    loss.backward()

# JSONL 格式示例 (train_stage2.jsonl):
{
    "model": "flux",
    "task_id": "portrait-001",
    "category": "C-Portrait",
    "scores": [
        {
            "subtask_idx": 0,
            "brf": 4.5,
            "veq_clarity": 4.0,
            "veq_realism": 3.5,
            "veq_aesthetic": 4.2,
            "veq_text": null,  # N/A
            "cns_edit": null,  # N/A
            "cns_set": 4.0,
            "label": 1  # Accept
        }
    ]
}
'''
    print(example_code_stage2)

    # ========== 旧架构（仍然支持）==========
    print("\n" + "=" * 80)
    print("【旧架构】V2/V3/V4（仍然支持，但建议迁移到统一架构）")
    print("=" * 80)

    # 示例4: 创建V2数据集（子任务级）
    print("\n示例4: V2数据集（子任务级评估）")
    print("-" * 80)
    example_code_v2 = '''
dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    version="v2",
    batch_size=8,
    shuffle_context=True  # 数据增强
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)
    # batch['labels']: (batch_size,)  # 单个值，不是数组
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
'''
    print(example_code_v2)

    # 示例5: 创建V3数据集（子任务级 + 图片压缩，推荐）
    print("\n示例5: V3数据集（子任务级 + 图片压缩）")
    print("-" * 80)
    example_code_v3 = '''
dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    version="v3",
    batch_size=8,
    shuffle_context=False  # V3推荐使用确定性压缩
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)  # 图片数量显著减少
    # batch['concept_labels']: (batch_size,)  # V3使用不同的字段名
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
'''
    print(example_code_v3)

    # 示例6: 创建V4数据集（多维度评分）
    print("\n示例6: V4数据集（7维度详细评分）")
    print("-" * 80)
    example_code_v4 = '''
dataloader = create_quality_eval_dataloader(
    train_jsonl_path="/path/to/train_scores.jsonl",
    bench_dir="/path/to/ServImage_Bench",
    dataset_dir="/path/to/ServImage_Dataset",
    processor=processor,
    version="v4",
    batch_size=4,
    shuffle=True
)

for batch in dataloader:
    # batch['input_ids']: (batch_size, seq_len)
    # batch['pixel_values']: (total_images, C, H, W)
    # batch['scores']: dict with 7 dimensions
    #   - 'brf': (batch_size,) - 0-5 or -1 for N/A
    #   - 'veq_clarity': (batch_size,) - 0-5
    #   - 'veq_realism': (batch_size,) - 0-5
    #   - 'veq_aesthetic': (batch_size,) - 0-5
    #   - 'veq_text': (batch_size,) - 0-5 or -1 for N/A
    #   - 'cns_edit': (batch_size,) - 1-5 or -1 for N/A
    #   - 'cns_set': (batch_size,) - 1-5 or -1 for N/A

    outputs = model(**batch)
    loss = compute_multidim_loss(outputs, batch['scores'])
    loss.backward()
'''
    print(example_code_v4)

    print("\n" + "=" * 80)
    print("架构对比总结:")
    print("-" * 80)
    print("【推荐】统一架构 (Base/Stage1/Stage2) - 两阶段训练管道:")
    print("  - 所有任务共享相同的数据加载和图片压缩逻辑")
    print("  - 只在任务 prompt 和输出格式上有区别")
    print("  - Base: Accept/Reject 预测（二分类）- 独立基础模型")
    print("  - Stage1: 7维度详细评分（BRF + VEQ 4维 + CNS 2维）- 第一阶段训练")
    print("  - Stage2: 基于Stage1特征的Accept/Reject判断 - 第二阶段训练")
    print("\n【旧架构】V2/V3/V4（已过时，仅保留兼容性）:")
    print("  - V2: 子任务级评估，单个标签（0/1）")
    print("  - V3: V2 + 动态网格图片压缩（减少视觉token数量）")
    print("  - V4: 多维度评分（7个维度详细评分）")
    print("\n注意：推荐使用统一架构（Base/Stage1/Stage2）")
    print("=" * 80)