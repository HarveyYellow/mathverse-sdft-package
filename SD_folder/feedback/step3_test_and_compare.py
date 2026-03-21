"""Step 3: Test v1 feedback on all-wrong problems and compare with v2 results.

Re-filters all-wrong using mathruler, then tests v1 feedbacks.
At the end, loads v2 test results and prints a comparison table.

Usage:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python feedback/step3_test_and_compare.py \
            --shard $SHARD --num_shards 4 &
    done
    wait
    # After all shards done, run summary:
    python feedback/step3_test_and_compare.py --summary_only
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

import argparse
import json
import glob
from mathruler.grader import extract_boxed_content, grade_answer

# ======================== Config ========================
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
EVAL_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
V1_FEEDBACK_DIR = os.path.join(BASE, "mathverse_feedback_v1")
V2_FEEDBACK_DIR = os.path.join(BASE, "mathverse_feedback_v2")
V1_TEST_DIR = os.path.join(BASE, "mathverse_feedback_v1_test")
V2_TEST_DIR = os.path.join(BASE, "mathverse_feedback_test")
NUM_ATTEMPTS = 8
MAX_NEW_TOKENS = 2048
MAX_PROMPT_TOKENS = 7000
CHUNK_SIZE = 16

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=4)
parser.add_argument("--summary_only", action="store_true")
args = parser.parse_args()


def check_answer(gold_str, model_output):
    extracted = extract_boxed_content(model_output)
    if not extracted:
        return False
    return grade_answer(extracted, gold_str)


def get_mathruler_all_wrong():
    """Re-score eval results with mathruler and return set of all-wrong indices."""
    eval_results = {}
    for f in sorted(glob.glob(f"{EVAL_DIR}/eval_shard*.jsonl")):
        with open(f) as fh:
            for line in fh:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in eval_results:
                    new_flags = [check_answer(item["answer"], r) for r in item["responses"]]
                    item["num_correct_in_8"] = sum(new_flags)
                    eval_results[idx] = item
    return {idx for idx, item in eval_results.items() if item["num_correct_in_8"] == 0}


def load_test_results(test_dir, valid_indices):
    """Load test results filtered to valid (mathruler all-wrong) indices."""
    results = {}
    for f in sorted(glob.glob(f"{test_dir}/test_shard*.jsonl")):
        with open(f) as fh:
            for line in fh:
                item = json.loads(line)
                if item.get("skipped"):
                    continue
                if item["idx"] in valid_indices:
                    results[item["idx"]] = item
    return results


def print_stats(name, results):
    if not results:
        print(f"  {name}: no results")
        return
    items = list(results.values())
    total = len(items)
    top1 = sum(1 for x in items if x["top1"])
    top8 = sum(1 for x in items if x["top8"])
    avg = sum(x["num_correct_in_8"] for x in items) / total
    print(f"  {name}: {total} problems | Top-1: {top1}/{total} ({top1*100/total:.1f}%) | "
          f"Top-8: {top8}/{total} ({top8*100/total:.1f}%) | Avg: {avg:.2f}/8")

    from collections import Counter
    dist = Counter(x["num_correct_in_8"] for x in items)
    for k in sorted(dist.keys()):
        pct = dist[k] * 100 / total
        print(f"    {k}/8: {dist[k]:4d} ({pct:5.1f}%)")


# ======================== Summary mode ========================
if args.summary_only:
    print("Loading mathruler all-wrong indices...")
    valid = get_mathruler_all_wrong()
    print(f"Mathruler all-wrong: {len(valid)}")

    print("\n=== V2 (General/Transferable) Feedback ===")
    v2_results = load_test_results(V2_TEST_DIR, valid)
    print_stats("V2", v2_results)

    print("\n=== V1 (Specific) Feedback ===")
    v1_results = load_test_results(V1_TEST_DIR, valid)
    print_stats("V1", v1_results)

    # Head-to-head on common problems
    common = set(v1_results.keys()) & set(v2_results.keys())
    if common:
        v1_better = sum(1 for idx in common if v1_results[idx]["num_correct_in_8"] > v2_results[idx]["num_correct_in_8"])
        v2_better = sum(1 for idx in common if v2_results[idx]["num_correct_in_8"] > v1_results[idx]["num_correct_in_8"])
        tie = len(common) - v1_better - v2_better
        print(f"\n=== Head-to-Head ({len(common)} common problems) ===")
        print(f"  V1 (specific) wins: {v1_better} ({v1_better*100/len(common):.1f}%)")
        print(f"  V2 (general) wins:  {v2_better} ({v2_better*100/len(common):.1f}%)")
        print(f"  Tie:                {tie} ({tie*100/len(common):.1f}%)")
    exit(0)

# ======================== Test mode (run inference) ========================
os.makedirs(V1_TEST_DIR, exist_ok=True)
OUTPUT_PATH = f"{V1_TEST_DIR}/test_shard{args.shard}.jsonl"

# Load v1 feedbacks
print(f"[Shard {args.shard}] Loading v1 feedbacks...")
feedback_data = {}
for f in sorted(glob.glob(f"{V1_FEEDBACK_DIR}/feedbacks_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            if item.get("skipped") or not item.get("feedbacks"):
                continue
            feedback_data[item["idx"]] = item

# Filter to mathruler all-wrong
valid_indices = get_mathruler_all_wrong()
all_indices = sorted(idx for idx in feedback_data if idx in valid_indices)
shard_indices = all_indices[args.shard::args.num_shards]

# Resume
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            completed.add(json.loads(line)["idx"])

pending = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] Total: {len(all_indices)}, shard: {len(shard_indices)}, pending: {len(pending)}")

if not pending:
    print(f"[Shard {args.shard}] Nothing to do!")
    exit(0)

# Load dataset & vLLM
from datasets import load_from_disk
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

ds = load_from_disk(DATASET_PATH + "/train")

print(f"[Shard {args.shard}] Loading vLLM...")
llm = LLM(
    model=MODEL_PATH, tensor_parallel_size=1, dtype="bfloat16",
    max_model_len=8192, trust_remote_code=True,
    gpu_memory_utilization=0.85, enforce_eager=True,
)
sampling_params = SamplingParams(n=1, temperature=1.0, top_p=0.9, max_tokens=MAX_NEW_TOKENS)

SYSTEM_PROMPT = ("Solve this problem. Put your final answer in \\boxed{}. "
                 "If the question provides multiple-choice options, only put the option letter in the box.")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer


def build_prompt_with_feedback(idx, feedback_text):
    sample = ds[idx]
    question = sample["question"]
    images = sample["images"]
    augmented = question.strip() + f"\n\n[Error Analysis from a previous attempt]:\n{feedback_text}"

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": augmented})

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


# ======================== Batch inference ========================
print(f"[Shard {args.shard}] Testing {len(pending)} problems × {NUM_ATTEMPTS} attempts...")
total_done = len(completed)

for chunk_start in range(0, len(pending), CHUNK_SIZE):
    chunk_indices = pending[chunk_start:chunk_start + CHUNK_SIZE]

    batch_prompts = []
    batch_meta = []

    for idx in chunk_indices:
        sample = ds[idx]
        fb_list = feedback_data[idx]["feedbacks"]

        for attempt in range(NUM_ATTEMPTS):
            fb_idx = attempt % len(fb_list)
            prompt_data = build_prompt_with_feedback(idx, fb_list[fb_idx])
            if prompt_data is None:
                continue
            batch_prompts.append(prompt_data)
            batch_meta.append({
                "idx": idx, "attempt": attempt, "feedback_idx": fb_idx,
                "answer": sample["answer"], "question": sample["question"],
            })

    if not batch_prompts:
        for idx in chunk_indices:
            sample = ds[idx]
            r = {"idx": idx, "question": sample["question"], "answer": sample["answer"],
                 "responses": [], "correct_flags": [], "feedback_indices": [],
                 "num_correct_in_8": 0, "top1": False, "top8": False, "skipped": True}
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_done += 1
        continue

    try:
        outputs = llm.generate(batch_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Chunk failed: {e}, skipping...")
        continue

    if len(outputs) != len(batch_meta):
        print(f"[Shard {args.shard}] WARNING: output/meta mismatch, skipping")
        continue

    results_by_idx = {}
    for i, output in enumerate(outputs):
        meta = batch_meta[i]
        idx = meta["idx"]
        response = output.outputs[0].text
        correct = check_answer(meta["answer"], response)

        if idx not in results_by_idx:
            results_by_idx[idx] = {
                "idx": idx, "question": meta["question"], "answer": meta["answer"],
                "responses": [], "correct_flags": [], "feedback_indices": [],
            }
        results_by_idx[idx]["responses"].append(response)
        results_by_idx[idx]["correct_flags"].append(correct)
        results_by_idx[idx]["feedback_indices"].append(meta["feedback_idx"])

    for idx in chunk_indices:
        if idx in results_by_idx:
            r = results_by_idx[idx]
            r["num_correct_in_8"] = sum(r["correct_flags"])
            r["top1"] = r["correct_flags"][0] if r["correct_flags"] else False
            r["top8"] = any(r["correct_flags"])
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}]")

print(f"\n[Shard {args.shard}] Done! {OUTPUT_PATH}")
