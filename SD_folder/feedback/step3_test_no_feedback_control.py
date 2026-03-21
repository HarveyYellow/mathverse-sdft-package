"""Control experiment: test all-wrong problems WITHOUT feedback (8 rollouts).

This establishes the true baseline for comparison with feedback experiments.
If a problem scores >0/8 here, it means the model CAN solve it sometimes
even without feedback — previous "all-wrong" label was due to sampling noise.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python feedback/step3_test_no_feedback_control.py
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

import argparse
import json
import glob
from vllm import LLM, SamplingParams
from datasets import load_from_disk
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content, grade_answer

# ======================== Config ========================
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
OUTPUT_DIR = os.path.join(BASE, "mathverse_no_feedback_control")
MATHRULER_INDICES_PATH = os.path.join(BASE, "mathruler_all_wrong_indices.json")
NUM_RESPONSES = 8
MAX_NEW_TOKENS = 2048
MAX_PROMPT_TOKENS = 7000
CHUNK_SIZE = 16

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/test_shard{args.shard}.jsonl"


def check_answer(gold_str, model_output):
    extracted = extract_boxed_content(model_output)
    if not extracted:
        return False
    return grade_answer(extracted, gold_str)


# Load mathruler all-wrong indices
with open(MATHRULER_INDICES_PATH) as f:
    all_wrong = sorted(json.load(f))

shard_indices = all_wrong[args.shard::args.num_shards]

# Resume
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            completed.add(json.loads(line)["idx"])

pending = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] Total: {len(all_wrong)}, shard: {len(shard_indices)}, pending: {len(pending)}")

if not pending:
    print("Nothing to do!")
    exit(0)

# Load dataset
ds = load_from_disk(DATASET_PATH + "/train")

# Load vLLM
print("Loading vLLM...")
llm = LLM(
    model=MODEL_PATH, tensor_parallel_size=1, dtype="bfloat16",
    max_model_len=8192, trust_remote_code=True,
    gpu_memory_utilization=0.85, enforce_eager=True,
)
sampling_params = SamplingParams(
    n=NUM_RESPONSES, temperature=1.0, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
)

SYSTEM_PROMPT = ("Solve this problem. Put your final answer in \\boxed{}. "
                 "If the question provides multiple-choice options, only put the option letter in the box.")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer


def build_prompt(idx):
    """Plain prompt, NO feedback."""
    sample = ds[idx]
    question = sample["question"]
    images = sample["images"]

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question.strip()})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
        return None

    img_inputs, _ = process_vision_info(messages)
    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs
    return {"prompt": prompt, "multi_modal_data": mm_data}


# Batch inference
print(f"Testing {len(pending)} problems × {NUM_RESPONSES} rollouts (no feedback)...")
total_done = len(completed)

for chunk_start in range(0, len(pending), CHUNK_SIZE):
    chunk_indices = pending[chunk_start:chunk_start + CHUNK_SIZE]

    batch_prompts = []
    batch_meta = []
    skipped = []

    for idx in chunk_indices:
        sample = ds[idx]
        prompt_data = build_prompt(idx)
        if prompt_data is None:
            skipped.append(idx)
            continue
        batch_prompts.append(prompt_data)
        batch_meta.append({"idx": idx, "answer": sample["answer"], "question": sample["question"]})

    # Write skipped
    for idx in skipped:
        sample = ds[idx]
        r = {"idx": idx, "question": sample["question"], "answer": sample["answer"],
             "responses": [], "correct_flags": [], "num_correct_in_8": 0,
             "top1": False, "top8": False, "skipped": True}
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_done += 1

    if not batch_prompts:
        continue

    try:
        outputs = llm.generate(batch_prompts, sampling_params)
    except Exception as e:
        print(f"Chunk failed: {e}, skipping...")
        continue

    if len(outputs) != len(batch_meta):
        print(f"WARNING: output/meta mismatch, skipping chunk")
        continue

    for i, output in enumerate(outputs):
        meta = batch_meta[i]
        responses = [o.text for o in output.outputs]
        correct_flags = [check_answer(meta["answer"], r) for r in responses]

        r = {
            "idx": meta["idx"],
            "question": meta["question"],
            "answer": meta["answer"],
            "responses": responses,
            "correct_flags": correct_flags,
            "num_correct_in_8": sum(correct_flags),
            "top1": correct_flags[0] if correct_flags else False,
            "top8": any(correct_flags),
        }
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_done += 1

    print(f"[{total_done}/{len(shard_indices)}]")

print(f"\nDone! Results: {OUTPUT_PATH}")
