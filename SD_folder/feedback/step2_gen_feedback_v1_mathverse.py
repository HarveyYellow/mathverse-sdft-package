"""Step 2 (v1): Generate 8 specific feedbacks per all-wrong problem (mathruler-scored).

Uses the original reflection prompt (specific error type + detail + correction hint).
Re-filters all-wrong problems using mathruler.grader for consistency.

Usage:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python feedback/step2_gen_feedback_v1_mathverse.py \
            --shard $SHARD --num_shards 4 &
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
from mathruler.grader import extract_boxed_content, grade_answer

# ======================== Config ========================
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
EVAL_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
OUTPUT_DIR = os.path.join(BASE, "mathverse_feedback_v1")
NUM_FEEDBACKS = 8
MAX_NEW_TOKENS = 512
CHUNK_SIZE = 2
MAX_PROMPT_TOKENS = 7000

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/feedbacks_shard{args.shard}.jsonl"

# ======================== Scoring ========================
def check_answer(gold_str, model_output):
    extracted = extract_boxed_content(model_output)
    if not extracted:
        return False
    return grade_answer(extracted, gold_str)

# ======================== Load eval results & re-score with mathruler ========================
print(f"[Shard {args.shard}] Loading and re-scoring eval results with mathruler...")
eval_results = {}
for f in sorted(glob.glob(f"{EVAL_DIR}/eval_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            idx = item["idx"]
            if idx not in eval_results:
                # Re-score with mathruler
                new_flags = [check_answer(item["answer"], r) for r in item["responses"]]
                item["correct_flags"] = new_flags
                item["num_correct_in_8"] = sum(new_flags)
                eval_results[idx] = item

wrong_items = {idx: r for idx, r in eval_results.items() if r["num_correct_in_8"] == 0}
all_wrong_indices = sorted(wrong_items.keys())
print(f"[Shard {args.shard}] Total eval: {len(eval_results)}, mathruler all-wrong: {len(all_wrong_indices)}")

# Shard & resume
shard_indices = all_wrong_indices[args.shard::args.num_shards]
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            completed.add(json.loads(line)["idx"])
    print(f"[Shard {args.shard}] Resuming: {len(completed)} done")

pending_indices = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] This shard: {len(shard_indices)}, pending: {len(pending_indices)}")

if not pending_indices:
    print(f"[Shard {args.shard}] Nothing to do!")
    exit(0)

# ======================== Load dataset ========================
print(f"[Shard {args.shard}] Loading dataset...")
ds = load_from_disk(DATASET_PATH + "/train")

# ======================== Load vLLM ========================
print(f"[Shard {args.shard}] Loading vLLM...")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 5},
)

sampling_params = SamplingParams(
    n=NUM_FEEDBACKS,
    temperature=0.7,
    top_p=0.95,
    max_tokens=MAX_NEW_TOKENS,
    frequency_penalty=0.3,
)

# ======================== V1 Prompt (Specific) ========================
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

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer


def build_prompt(idx):
    sample = ds[idx]
    question = sample["question"]
    answer = sample["answer"]
    images = sample["images"]
    eval_item = wrong_items[idx]

    student_resp = min(eval_item["responses"], key=len)
    if len(student_resp) < 200:
        student_resp = eval_item["responses"][0]
    if len(student_resp) > 1500:
        student_resp = student_resp[:1500] + "\n... [truncated]"

    text = REFLECTION_TEMPLATE.format(
        problem=question.strip(),
        student_response=student_resp,
        gold_answer=answer,
    )

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Filter overlong
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
        return None, None

    img_inputs, _ = process_vision_info(messages)
    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs

    return (
        {"prompt": prompt, "multi_modal_data": mm_data},
        {"idx": idx, "question": question, "gold_answer": answer},
    )


# ======================== Generate ========================
print(f"[Shard {args.shard}] Generating {NUM_FEEDBACKS} feedbacks for {len(pending_indices)} problems...")
total_done = len(completed)

for chunk_start in range(0, len(pending_indices), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(pending_indices))
    chunk_indices = pending_indices[chunk_start:chunk_end]

    chunk_prompts = []
    chunk_meta = []
    skipped_indices = []

    for idx in chunk_indices:
        prompt, meta = build_prompt(idx)
        if prompt is None:
            skipped_indices.append(idx)
            continue
        chunk_prompts.append(prompt)
        chunk_meta.append(meta)

    # Write skipped
    for idx in skipped_indices:
        sample = ds[idx]
        result = {
            "idx": idx, "question": sample["question"], "gold_answer": sample["answer"],
            "feedbacks": [], "raw_outputs": [], "skipped": True,
        }
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        total_done += 1

    if not chunk_prompts:
        continue

    try:
        outputs = llm.generate(chunk_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Chunk failed: {e}, skipping...")
        continue

    if len(outputs) != len(chunk_meta):
        print(f"[Shard {args.shard}] WARNING: outputs/meta mismatch, skipping chunk")
        continue

    for i, output in enumerate(outputs):
        meta = chunk_meta[i]
        feedbacks = [o.text for o in output.outputs]

        result = {
            "idx": meta["idx"],
            "question": meta["question"],
            "gold_answer": meta["gold_answer"],
            "feedbacks": feedbacks,
            "raw_outputs": feedbacks,
        }
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}]")

print(f"\n[Shard {args.shard}] Done! Results: {OUTPUT_PATH}")
