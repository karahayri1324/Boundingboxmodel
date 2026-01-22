#!/usr/bin/env python3
"""
Test REAP V8 Pruned Qwen3-VL Model in Docker
=============================================
Tests the newly pruned model with multimodal bbox detection.
"""

import os
import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Paths (inside docker container)
BASE_DIR = "/app/data"
MODEL_PATH = f"{BASE_DIR}/Qwen3-VL-235B-A22B-REAP-Pruned"
CALIBRATION_DATA = f"{BASE_DIR}/reap_calibration_data/calibration_data.json"
IMAGES_DIR = f"{BASE_DIR}/reap_calibration_data/images"
OUTPUT_DIR = Path(f"{BASE_DIR}/testresults_reap_v8")

# Test settings
NUM_TESTS = 20
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size=16):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    return ImageFont.load_default()


def parse_bbox(response: str) -> list:
    patterns = [
        r'\{"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\}',
        r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        r'bbox_2d["\s:]*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
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

    for i in range(4):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

    font = get_font(14)
    label = region[:50] + "..." if len(region) > 50 else region

    try:
        tb = draw.textbbox((x1, y1-22), label, font=font)
    except:
        tb = (x1, y1-22, x1+len(label)*8, y1-5)

    draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=color)
    draw.text((x1, y1-22), label, fill=(0, 0, 0), font=font)

    return img


def main():
    print("=" * 70)
    print("  Testing REAP V8 Pruned Qwen3-VL Model")
    print("  Bounding Box Detection with Vision Input")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # Check config
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nModel Info:")
    print(f"  Path: {MODEL_PATH}")
    print(f"  Experts: {config.get('text_config', {}).get('num_experts', 'N/A')}")
    print(f"  REAP Pruned: {config.get('reap_pruned', False)}")
    print(f"  Original Experts: {config.get('reap_original_experts', 'N/A')}")

    # Load data
    print(f"\nLoading calibration data...")
    with open(CALIBRATION_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(RANDOM_SEED)
    test_samples = random.sample(data, min(NUM_TESTS, len(data)))
    print(f"Selected {len(test_samples)} samples for testing")

    # Load vLLM with multimodal support
    print(f"\nLoading model with vLLM...")

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
        stop=["<|im_end|>", "\n\n", "```"],
    )

    print("Model loaded successfully!")

    # Process each sample with image
    print("\nRunning inference with images...")
    results = []
    successful = 0
    failed = 0

    for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
        image_path = os.path.join(IMAGES_DIR, os.path.basename(sample['image']))
        regions = sample.get('regions', [])

        if not regions or not os.path.exists(image_path):
            continue

        region = random.choice(regions)

        # Load image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Create multimodal prompt
        prompt = f'''<|im_start|>system
You are a visual detection assistant. Analyze the image and find the specified region. Return bounding box coordinates in JSON format with values normalized between 0-1000.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Find "{region}" in this image and return the bounding box.
Format: {{"bbox_2d": [x1, y1, x2, y2]}}<|im_end|>
<|im_start|>assistant
{{"bbox_2d": ['''

        try:
            # Generate with image
            outputs = llm.generate(
                [{
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }],
                sampling_params,
            )

            response = outputs[0].outputs[0].text.strip()
            full_response = '{"bbox_2d": [' + response
            bbox = parse_bbox(full_response)

            if bbox:
                bbox_px = norm_to_pixels(bbox, w, h)
                result_img = draw_result(image, bbox_px, region)
                out_path = OUTPUT_DIR / f"test_{idx:03d}_{sample['id']}.png"
                result_img.save(out_path)

                results.append({
                    "id": sample['id'],
                    "region": region,
                    "response": full_response,
                    "bbox_norm": bbox,
                    "bbox_px": bbox_px,
                    "status": "success",
                })
                successful += 1
                print(f"  [OK] {region[:30]}... -> {bbox}")
            else:
                draw = ImageDraw.Draw(image)
                font = get_font(16)
                draw.text((10, 10), f"FAILED: {region[:40]}", fill=(255, 0, 0), font=font)
                draw.text((10, 35), f"Resp: {response[:50]}", fill=(255, 100, 0), font=get_font(12))

                out_path = OUTPUT_DIR / f"test_{idx:03d}_{sample['id']}_FAIL.png"
                image.save(out_path)

                results.append({
                    "id": sample['id'],
                    "region": region,
                    "response": full_response,
                    "status": "failed",
                })
                failed += 1
                print(f"  [FAIL] {region[:30]}... -> {response[:30]}...")

        except Exception as e:
            print(f"  [ERROR] {region[:30]}... -> {str(e)[:50]}")
            results.append({
                "id": sample['id'],
                "region": region,
                "error": str(e),
                "status": "error",
            })
            failed += 1

    # Save results
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": MODEL_PATH,
            "version": "REAP V8 Pruned",
            "experts": config.get('text_config', {}).get('num_experts', 'N/A'),
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "rate": f"{successful/len(results)*100:.1f}%" if results else "0%",
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  TEST COMPLETE!")
    print("=" * 70)
    print(f"""
  Model: REAP V8 Pruned Qwen3-VL
  Experts: {config.get('text_config', {}).get('num_experts', 'N/A')} (from 128)
  Mode: MULTIMODAL (Image + Text)

  Total: {len(results)}
  Success: {successful}
  Failed: {failed}
  Rate: {successful/len(results)*100:.1f}%

  Results saved to: {OUTPUT_DIR}
""")


if __name__ == "__main__":
    main()
