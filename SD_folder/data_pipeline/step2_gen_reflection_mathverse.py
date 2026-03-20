"""Step 2: Generate 8 reflections per all-wrong problem.

Two modes via --truncate flag:
  --truncate       : truncate student response to 1500 chars (default)
  --no-truncate    : keep full student response

Uses vLLM with data parallel (1 GPU per shard).

Usage:
    # Truncated version:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python step2_gen_reflection_mathverse.py \
            --shard $SHARD --num_shards 4 --truncate &
    done
    wait

    # No-truncate version:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python step2_gen_reflection_mathverse.py \
            --shard $SHARD --num_shards 4 --no-truncate &
    done
    wait
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

import argparse
import json
import glob
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ======================== Config ========================
MODEL_PATH = "/lscratch/jh19696/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/scratch/jh19696/Self-Distillation/data/sdft_mathverse"
EVAL_DIR = "/scratch/jh19696/Self-Distillation/mathverse_eval_results_vllm"
NUM_REFLECTIONS = 8
MAX_NEW_TOKENS = 512
CHUNK_SIZE = 2

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
parser.add_argument("--truncate", action="store_true", default=False,
                    help="Truncate student response to 1500 chars")
parser.add_argument("--no-truncate", action="store_true", default=False,
                    help="Keep full student response (no truncation)")
args = parser.parse_args()

# Determine truncation mode
if args.no_truncate:
    do_truncate = False
elif args.truncate:
    do_truncate = True
else:
    do_truncate = True  # default: truncate

# Output dir based on mode
if do_truncate:
    OUTPUT_DIR = "/scratch/jh19696/Self-Distillation/mathverse_reflection_n8_truncated"
else:
    OUTPUT_DIR = "/scratch/jh19696/Self-Distillation/mathverse_reflection_n8_full"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/reflection_shard{args.shard}.jsonl"

print(f"[Shard {args.shard}] Truncation mode: {'ON (1500 chars)' if do_truncate else 'OFF (full response)'}")
print(f"[Shard {args.shard}] Output dir: {OUTPUT_DIR}")

# ======================== Load eval results ========================
print(f"[Shard {args.shard}] Loading eval results...")
eval_results = {}
for f in sorted(glob.glob(f"{EVAL_DIR}/eval_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            if item["idx"] not in eval_results:
                eval_results[item["idx"]] = item

# Filter: all 8 wrong
wrong_items = {idx: r for idx, r in eval_results.items() if r["num_correct_in_8"] == 0}
all_wrong_indices = sorted(wrong_items.keys())
print(f"[Shard {args.shard}] Total eval results: {len(eval_results)}, all-wrong problems: {len(all_wrong_indices)}")

# Shard
shard_indices = all_wrong_indices[args.shard::args.num_shards]

# Resume support
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            completed.add(item["idx"])
    print(f"[Shard {args.shard}] Resuming: {len(completed)} already done")

pending_indices = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] This shard={len(shard_indices)}, pending={len(pending_indices)}")

if not pending_indices:
    print(f"[Shard {args.shard}] Nothing to do!")
    exit(0)

# ======================== Load dataset ========================
print(f"[Shard {args.shard}] Loading dataset...")
ds = load_from_disk(DATASET_PATH + "/train")

# ======================== Load vLLM ========================
print(f"[Shard {args.shard}] Loading vLLM model...")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.6,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    n=NUM_REFLECTIONS,
    temperature=0.7,
    top_p=0.95,
    max_tokens=MAX_NEW_TOKENS,
    frequency_penalty=0.3,
)

# ======================== Build prompts ========================
REFLECTION_TEMPLATE = """You are a math teacher reviewing a student's wrong solution. Be brief and precise — your entire response must be under 150 words.

## Problem
{problem}

## Student's Solution (WRONG)
{student_response}

## Correct Answer
{gold_answer}

## Task
Identify the specific error in 3 lines:
1. **Error Type** (one of: Image Misreading / Theorem Misapplication / Calculation Error / Reasoning Loop / Variable Misassignment / Diagram Relationship Error / Option Mapping Error)
2. **Error Detail**: One sentence pinpointing exactly where the reasoning went wrong.
3. **Correction Hint**: One sentence telling the student how to fix it.

Do NOT re-solve the problem. Do NOT repeat yourself. Keep it short."""

print(f"[Shard {args.shard}] Building prompts...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

batch_prompts = []
batch_indices = []
batch_meta = []

for idx in pending_indices:
    sample = ds[idx]
    question = sample["question"]
    answer = sample["answer"]
    images = sample["images"]
    eval_item = wrong_items[idx]

    # Pick the shortest wrong response
    student_resp = min(eval_item["responses"], key=len)
    if len(student_resp) < 200:
        student_resp = eval_item["responses"][0]

    # Truncate if enabled
    if do_truncate and len(student_resp) > 1500:
        student_resp = student_resp[:1500] + "\n... [truncated]"

    reflection_text = REFLECTION_TEMPLATE.format(
        problem=question.strip(),
        student_response=student_resp,
        gold_answer=answer,
    )

    # Build multimodal message
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": reflection_text})

    messages = [
        {"role": "user", "content": content},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, _ = process_vision_info(messages)

    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs

    batch_prompts.append({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })
    batch_indices.append(idx)
    batch_meta.append({
        "idx": idx,
        "question": question,
        "gold_answer": answer,
    })

# ======================== Batch Inference ========================
print(f"[Shard {args.shard}] Generating {NUM_REFLECTIONS} reflections each for {len(batch_prompts)} problems (chunk_size={CHUNK_SIZE})...")
total_done = len(completed)

for chunk_start in range(0, len(batch_prompts), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(batch_prompts))
    chunk_prompts = batch_prompts[chunk_start:chunk_end]
    chunk_meta = batch_meta[chunk_start:chunk_end]

    try:
        outputs = llm.generate(chunk_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Chunk {chunk_start//CHUNK_SIZE + 1} FAILED: {e}")
        print(f"[Shard {args.shard}] Falling back to one-by-one...")
        for i, (prompt, meta) in enumerate(zip(chunk_prompts, chunk_meta)):
            try:
                single_out = llm.generate([prompt], sampling_params)
                reflections = [o.text for o in single_out[0].outputs]
            except Exception as e2:
                print(f"[Shard {args.shard}]   idx={meta['idx']} SKIPPED: {e2}")
                reflections = ["[GENERATION_ERROR]"] * NUM_REFLECTIONS

            result = {
                "idx": meta["idx"],
                "question": meta["question"],
                "gold_answer": meta["gold_answer"],
                "reflections": reflections,
            }
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            total_done += 1

        print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] Chunk {chunk_start//CHUNK_SIZE + 1} done (with fallback)")
        continue

    for i, output in enumerate(outputs):
        reflections = [o.text for o in output.outputs]
        meta = chunk_meta[i]

        result = {
            "idx": meta["idx"],
            "question": meta["question"],
            "gold_answer": meta["gold_answer"],
            "reflections": reflections,
        }

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] Chunk {chunk_start//CHUNK_SIZE + 1} done")

print(f"\n[Shard {args.shard}] Done! Results saved to {OUTPUT_PATH}")
