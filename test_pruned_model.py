#!/usr/bin/env python3
"""
Test REAP V11 Pruned Model
==========================

Tests the pruned Qwen3-VL model with multimodal bbox detection.
"""

import os
import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# Paths
BASE_DIR = os.environ.get("REAP_DATA_DIR", "/app/data")
MODEL_PATH = os.environ.get(
    "REAP_PRUNED_OUTPUT_PATH",
    f"{BASE_DIR}/Qwen3-VL-235B-REAP-Pruned-V11"
)
CALIBRATION_DATA = os.environ.get(
    "REAP_CALIBRATION_PATH",
    f"{BASE_DIR}/reap_calibration_data/calibration_data.json"
)
OUTPUT_DIR = Path(f"{BASE_DIR}/test_results_v11")

NUM_TESTS = 20
SEED = 42


def get_font(size=16):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                continue
    return ImageFont.load_default()


def parse_bbox(response: str) -> list:
    patterns = [
        r'\{"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\}',
        r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            coords = [int(match.group(i)) for i in range(1, 5)]
            if all(0 <= c <= 1000 for c in coords):
                return coords
    return None


def norm_to_pixels(bbox, w, h):
    return [
        int(bbox[0] / 1000 * w),
        int(bbox[1] / 1000 * h),
        int(bbox[2] / 1000 * w),
        int(bbox[3] / 1000 * h),
    ]


def draw_result(image, bbox_px, region, color=(0, 255, 0)):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_px
    
    for i in range(3):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
    
    font = get_font(14)
    label = region[:40] + "..." if len(region) > 40 else region
    draw.text((x1, y1-20), label, fill=color, font=font)
    
    return img


def main():
    print("="*70)
    print("  Testing REAP V11 Pruned Model")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    text_config = config.get('text_config', config)
    num_experts = text_config.get('num_experts', 'N/A')
    reap_info = config.get('reap_pruning', {})
    
    print(f"\nModel: {MODEL_PATH}")
    print(f"Experts: {num_experts}")
    print(f"REAP Version: {reap_info.get('version', 'unknown')}")
    print(f"Original Experts: {reap_info.get('original_experts', 'N/A')}")
    
    # Load data
    if not os.path.exists(CALIBRATION_DATA):
        print(f"ERROR: Calibration data not found: {CALIBRATION_DATA}")
        return
    
    with open(CALIBRATION_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.seed(SEED)
    test_samples = random.sample(data, min(NUM_TESTS, len(data)))
    print(f"\nTest samples: {len(test_samples)}")
    
    # Load vLLM
    print("\nLoading model with vLLM...")
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="auto",
        limit_mm_per_prompt={"image": 1},
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=150,
        stop=["<|im_end|>", "\n\n"],
    )
    
    print("Model loaded!")
    
    # Test
    print("\nRunning inference...")
    results = []
    success = 0
    fail = 0
    
    images_dir = os.path.dirname(CALIBRATION_DATA)
    if 'images' not in images_dir:
        images_dir = os.path.join(os.path.dirname(CALIBRATION_DATA), 'images')
    
    for idx, sample in enumerate(tqdm(test_samples, desc="Testing")):
        image_path = os.path.join(images_dir, os.path.basename(sample.get('image', '')))
        regions = sample.get('regions', [])
        
        if not regions or not os.path.exists(image_path):
            continue
        
        region = random.choice(regions)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        prompt = f'''<|im_start|>system
You are a visual detection assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Find "{region}" and return bounding box.
Format: {{"bbox_2d": [x1, y1, x2, y2]}}<|im_end|>
<|im_start|>assistant
{{"bbox_2d": ['''
        
        try:
            outputs = llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": image}}],
                sampling_params,
            )
            
            response = outputs[0].outputs[0].text.strip()
            full_response = '{"bbox_2d": [' + response
            bbox = parse_bbox(full_response)
            
            if bbox:
                bbox_px = norm_to_pixels(bbox, w, h)
                result_img = draw_result(image, bbox_px, region)
                result_img.save(OUTPUT_DIR / f"test_{idx:03d}_OK.png")
                
                results.append({
                    "id": sample.get('id', idx),
                    "region": region,
                    "bbox": bbox,
                    "status": "success",
                })
                success += 1
                print(f"  [OK] {region[:30]}... -> {bbox}")
            else:
                image.save(OUTPUT_DIR / f"test_{idx:03d}_FAIL.png")
                results.append({
                    "id": sample.get('id', idx),
                    "region": region,
                    "response": response[:100],
                    "status": "failed",
                })
                fail += 1
                print(f"  [FAIL] {region[:30]}... -> {response[:50]}")
                
        except Exception as e:
            print(f"  [ERROR] {str(e)[:50]}")
            fail += 1
    
    # Save results
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump({
            "model": MODEL_PATH,
            "experts": num_experts,
            "total": len(results),
            "success": success,
            "fail": fail,
            "rate": f"{success/(success+fail)*100:.1f}%" if (success+fail) > 0 else "0%",
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"  Success: {success}, Failed: {fail}")
    print(f"  Rate: {success/(success+fail)*100:.1f}%" if (success+fail) > 0 else "N/A")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
