"""Step 3: Test whether feedback improves model performance on all-wrong problems.

For each all-wrong problem (1067 total):
  - 8 attempts, each with a different feedback injected into the prompt
  - Feedbacks are evenly distributed across attempts
  - Score with mathruler.grader (same as GRPO training)

Uses vLLM with data parallel (1 GPU per shard).

Usage:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python feedback/step3_test_feedback_mathverse.py \
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
FEEDBACK_DIR = os.path.join(BASE, "mathverse_feedback_v2")
OUTPUT_DIR = os.path.join(BASE, "mathverse_feedback_test")
NUM_ATTEMPTS = 8
MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 1  # problems per chunk (each expands to 8 prompts); keep at 1 to isolate failures

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/test_shard{args.shard}.jsonl"

# ======================== Answer Check ========================
def check_answer(gold_str, model_output):
    extracted = extract_boxed_content(model_output)
    if not extracted:
        return False
    return grade_answer(extracted, gold_str)

# ======================== Resume ========================
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            completed.add(item["idx"])
    print(f"[Shard {args.shard}] Resuming: {len(completed)} already completed")

# ======================== Load Feedbacks ========================
print(f"[Shard {args.shard}] Loading feedbacks...")
feedback_data = {}
for fpath in sorted(glob.glob(f"{FEEDBACK_DIR}/feedbacks_shard*.jsonl")):
    with open(fpath) as fh:
        for line in fh:
            item = json.loads(line)
            feedback_data[item["idx"]] = item

all_indices = sorted(feedback_data.keys())
shard_indices = all_indices[args.shard::args.num_shards]
pending_indices = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] Total feedback problems: {len(all_indices)}, "
      f"this shard: {len(shard_indices)}, pending: {len(pending_indices)}")

if not pending_indices:
    print(f"[Shard {args.shard}] Nothing to do!")
    exit(0)

# ======================== Load Dataset ========================
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
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    n=1,
    temperature=1.0,
    top_p=0.9,
    max_tokens=MAX_NEW_TOKENS,
)

# ======================== Prompt Template ========================
SYSTEM_PROMPT = ("Solve this problem. Put your final answer in \\boxed{}. "
                 "If the question provides multiple-choice options, only put the option letter in the box.")

processor = AutoProcessor.from_pretrained(MODEL_PATH)


def build_prompt_with_feedback(idx, feedback_text):
    """Build a prompt with feedback injected."""
    sample = ds[idx]
    question = sample["question"]
    images = sample["images"]

    augmented_question = (question.strip() +
                          f"\n\n[Error Analysis from a previous attempt]:\n{feedback_text}")

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": augmented_question})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, _ = process_vision_info(messages)

    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs

    return {"prompt": prompt, "multi_modal_data": mm_data}


# ======================== Batch Inference ========================
print(f"[Shard {args.shard}] Testing {len(pending_indices)} problems × {NUM_ATTEMPTS} attempts "
      f"(chunk_size={CHUNK_SIZE})...")
total_done = len(completed)

for chunk_start in range(0, len(pending_indices), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(pending_indices))
    chunk_indices = pending_indices[chunk_start:chunk_end]

    # Build 8 prompts per problem, each with a different feedback
    batch_prompts = []
    batch_meta = []  # track (idx, attempt, feedback_idx)

    for idx in chunk_indices:
        sample = ds[idx]
        answer = sample["answer"]
        question = sample["question"]
        fb_item = feedback_data[idx]
        fb_list = fb_item["feedbacks"]

        for attempt in range(NUM_ATTEMPTS):
            fb_idx = attempt % len(fb_list)
            fb_text = fb_list[fb_idx]

            prompt_data = build_prompt_with_feedback(idx, fb_text)
            batch_prompts.append(prompt_data)
            batch_meta.append({
                "idx": idx,
                "attempt": attempt,
                "feedback_idx": fb_idx,
                "answer": answer,
                "question": question,
            })

    try:
        outputs = llm.generate(batch_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Chunk failed: {e}, skipping...")
        continue

    # Collect results by idx
    results_by_idx = {}
    if len(outputs) != len(batch_meta):
        print(f"[Shard {args.shard}] WARNING: outputs ({len(outputs)}) != batch_meta ({len(batch_meta)}), skipping chunk")
        continue
    for i, output in enumerate(outputs):
        meta = batch_meta[i]
        idx = meta["idx"]

        if output is None:
            response = "[GENERATION_ERROR]"
            correct = False
        else:
            response = output.outputs[0].text
            correct = check_answer(meta["answer"], response)

        if idx not in results_by_idx:
            results_by_idx[idx] = {
                "idx": idx,
                "question": meta["question"],
                "answer": meta["answer"],
                "responses": [],
                "correct_flags": [],
                "feedback_indices": [],
            }

        results_by_idx[idx]["responses"].append(response)
        results_by_idx[idx]["correct_flags"].append(correct)
        results_by_idx[idx]["feedback_indices"].append(meta["feedback_idx"])

    # Write results
    for idx in chunk_indices:
        if idx in results_by_idx:
            r = results_by_idx[idx]
            r["num_correct_in_8"] = sum(r["correct_flags"])
            r["top1"] = r["correct_flags"][0]
            r["top8"] = any(r["correct_flags"])

            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

            total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] "
          f"Chunk {chunk_start // CHUNK_SIZE + 1} done")

print(f"\n[Shard {args.shard}] Done! Results saved to {OUTPUT_PATH}")
