from transformers import AutoProcessor
from datasets import load_dataset
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
import re
import os
import json
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from vllm import LLM, SamplingParams

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(42)

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "./GUI-RC"
TEMPERATURE = 0.7
SAMPLE_NUM = 64
BATCH_SIZE = 1024
POINT_EXPAND_SIZE = 50

def extract_answer(content, point_expand_size=50):
    numbers = re.findall(r'\d+(?:\.\d+)?', content)
    expand = point_expand_size / 2
    if len(numbers) == 4:
        # If 4 numbers found, return as bbox [x1, y1, x2, y2]
        return [float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])]
    elif len(numbers) == 2:
        # If 2 numbers found, treat as point [x, y] and expand to bbox
        x, y = float(numbers[0]), float(numbers[1])
        return [x - expand, y - expand, x + expand, y + expand]
    else:
        return [0, 0, 0, 0]
            
def point_in_bbox(point, bbox):
    """Checks if a point is within a bounding box."""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2

QUESTION_TEMPLATE = (
    "You are a GUI analysis expert. Based on the screenshot, your task is to outline the UI element that best matches the instruction:\n"
    "'{instruction}'\n"
    "You MUST output exactly one bounding box in the format [x1, y1, x2, y2] strictly. Do not include any explanations, descriptions, or additional text.\n"
    "**Output format: [x1, y1, x2, y2]**"
)

def load_jsonl_data(file_path):
    """加载JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 构造图像路径
            if 'screenspot_pro' in file_path:
                image_path = f"data/screenspot_pro/images/{item['image']}"
            elif 'screenspot_v2' in file_path:
                image_path = f"data/screenspot_v2/images/{item['image']}"
            else:
                image_path = f"data/screenspot/images/{item['image']}"
            
            # 转换为与原始数据集格式兼容的结构
            converted_item = {
                'image_path': image_path,
                'instruction': item['instruction'],
                'bbox': item['bbox'],
                'data_source': item['data_source'],
                'data_type': item['data_type'],
                'image_size': item['image_size']
            }
            data.append(converted_item)
    return data

# --- 数据集文件路径配置 ---
dataset_files = {
    "screenspot": "data/screenspot/screenspot.jsonl",
    "screenspot_v2": "data/screenspot_v2/screenspot_v2.jsonl", 
    "screenspot_pro": "data/screenspot_pro/screenspot_pro.jsonl"
}

# --- vLLM and Processor Initialization ---
# print("Initializing vLLM engine...")
# Note: tensor_parallel_size can be adjusted based on the number of available GPUs.
# trust_remote_code=True is required for many Hugging Face models.
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=24576 # Adjust if necessary based on model's context length
)

print("Initializing processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

# Define sampling parameters for vLLM
# We use n=K to get K samples per prompt in a single call.
sampling_params = SamplingParams(
    n=SAMPLE_NUM,
    temperature=TEMPERATURE,
    top_p=0.95,
    max_tokens=1024 # Corresponds to max_new_tokens
)

# --- Data Loading ---

for dataset_name, file_path in dataset_files.items():
    print(f"\n📊 评测数据集: {dataset_name}")
    data = load_jsonl_data(file_path)

    results = []
    correct_number = 0
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for batch_start in tqdm(range(0, len(data), BATCH_SIZE), desc=f"Evaluating with vLLM, Dataset: {dataset_name}"):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_data = data[batch_start:batch_end]

        batch_images = [Image.open(example["image_path"]).convert("RGB") for example in batch_data]
        batch_instructions = [example["instruction"] for example in batch_data]
        batch_gt_bboxes = [example["bbox"] for example in batch_data]
        
        # 1. Prepare inputs for vLLM
        inputs = []
        for i in range(len(batch_data)):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": batch_images[i]},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(instruction=batch_instructions[i])}
                ]
            }]
            
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": batch_images[i]}
            })

        # 2. Generate outputs with vLLM
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params
        )
        
        # 3. Process the results
        for i, request_output in enumerate(outputs):
            original_index = batch_start + i
            example = data[original_index]
            gt_bbox = batch_gt_bboxes[i]
            W, H = batch_images[i].size
            gt_bbox = [gt_bbox[0] * W, gt_bbox[1] * H, gt_bbox[2] * W, gt_bbox[3] * H]
            grid = torch.zeros((H, W), dtype=torch.int32)

            sampled_bboxes = []
            for completion_output in request_output.outputs:
                output_text = completion_output.text
                bbox = extract_answer(output_text, POINT_EXPAND_SIZE)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                sampled_bboxes.append([x1, y1, x2, y2])
                grid[y1:y2, x1:x2] += 1

            correct = 0
            consensus_region = None
            max_votes = grid.max().item()
            if max_votes > 0:
                # 找到所有票数最高的区域
                max_vote_mask = (grid == max_votes)
                
                # 使用label标记连通区域
                labeled_array, num_features = label(max_vote_mask)
                
                if num_features > 0:
                    # 使用find_objects找到每个区域的边界框
                    region_slices = find_objects(labeled_array)
                    
                    max_area = 0
                    consensus_region_center = None
                    
                    for region_id, slice_tuple in enumerate(region_slices):
                        if slice_tuple is not None:  # 确保区域存在
                            y_slice, x_slice = slice_tuple
                            # 计算区域面积
                            region_mask = (labeled_array == region_id + 1)
                            area = region_mask.sum()
                            
                            if area > max_area:
                                max_area = area
                                # 计算区域中心点
                                center_y = (y_slice.start + y_slice.stop) / 2
                                center_x = (x_slice.start + x_slice.stop) / 2
                                consensus_region_center = (center_x, center_y)
                                consensus_region = [x_slice.start, y_slice.start, x_slice.stop, y_slice.stop]

                    if consensus_region_center is not None:
                        if point_in_bbox(consensus_region_center, gt_bbox):
                            correct = 1

            data_source = example["data_source"]
            data_type = example["data_type"]
            group_stats[(data_source, data_type)]["total"] += 1
            group_stats[(data_source, data_type)]["correct"] += correct
            correct_number += correct

            result = {
                'id': original_index,
                'instruction': batch_instructions[i],
                'ground_truth': gt_bbox,
                'sampled_bboxes': sampled_bboxes,
                'consensus_region': consensus_region,
                'correct': correct,
                'data_source': data_source,
                'data_type': data_type
            }
            results.append(result)

    # --- Final Accuracy Calculation and Reporting ---
    total_samples = len(data)
    accuracy = (correct_number / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    group_accuracy = {}
    for (ds, dt), stat in group_stats.items():
        acc = (stat["correct"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        group_accuracy[f"{ds}_{dt}"] = round(acc, 2)
        print(f"{ds}_{dt}: {acc:.2f}%")

    # --- Save Results to JSON ---
    MODEL = MODEL_PATH.split("/")[-2] + "/" + MODEL_PATH.split("/")[-1]
    DATASET = dataset_name
    OUTPUT_PATH = f"{OUTPUT_DIR}/{MODEL}/{DATASET}_sample{SAMPLE_NUM}_temp{TEMPERATURE}.json"
    if not os.path.exists(f"{OUTPUT_DIR}/{MODEL}"):
        os.makedirs(f"{OUTPUT_DIR}/{MODEL}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            'accuracy': round(accuracy, 2),
            'group_accuracy': group_accuracy,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")
