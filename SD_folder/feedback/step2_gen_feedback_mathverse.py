"""Step 2 (v2): Generate 8 transferable feedbacks per all-wrong problem.

New prompt design:
  - Model first thinks (in <think> tags, discarded later)
  - Then gives transferable feedback (in <feedback> tags, extracted for training)
  - Feedback focuses on: what to pay attention to + how to avoid this type of error
  - No problem-specific details (numbers, options, variable names)

Uses vLLM with data parallel (1 GPU per shard).

Usage:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python step2_gen_feedback_mathverse.py \
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
import re
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ======================== Config ========================
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
EVAL_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
NUM_FEEDBACKS = 8
MAX_NEW_TOKENS = 512
CHUNK_SIZE = 2

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
parser.add_argument("--truncate", action="store_true", default=True,
                    help="Truncate student response to 1500 chars (default)")
parser.add_argument("--no-truncate", action="store_true", default=False,
                    help="Keep full student response")
args = parser.parse_args()

do_truncate = not args.no_truncate

OUTPUT_DIR = os.path.join(BASE, "mathverse_feedback_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/feedbacks_shard{args.shard}.jsonl"

print(f"[Shard {args.shard}] Truncation: {'ON (1500 chars)' if do_truncate else 'OFF'}")
print(f"[Shard {args.shard}] Output: {OUTPUT_PATH}")

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
print(f"[Shard {args.shard}] Total eval results: {len(eval_results)}, all-wrong: {len(all_wrong_indices)}")

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
    limit_mm_per_prompt={"image": 5},
)

sampling_params = SamplingParams(
    n=NUM_FEEDBACKS,
    temperature=0.7,
    top_p=0.95,
    max_tokens=MAX_NEW_TOKENS,
    frequency_penalty=0.3,
)

# ======================== New Prompt ========================
FEEDBACK_TEMPLATE = """You are a math teacher. A student made an error on the following problem.

## Problem
{problem}

## Student's Wrong Solution
{student_response}

## Correct Answer
{gold_answer}

## Task
First, think step by step about what type of mistake the student made.
Then, provide a short, transferable piece of feedback that tells the student:
- What they should pay attention to when facing this type of problem
- How to check and correct this kind of mistake

## Requirements
- The feedback MUST be transferable to other problems of the same type
- Do NOT mention any specific numbers, options, variable names, or details from this problem
- Do NOT re-solve the problem or reveal the correct answer
- Write the feedback as if you are giving a general study tip, not correcting this one problem

## Output Format
<think>
(your reasoning about the student's error)
</think>
<feedback>
(2-3 sentences: what to pay attention to + how to avoid this type of error in the future)
</feedback>"""


def extract_feedback(text):
    """Extract content from <feedback> tags."""
    m = re.search(r"<feedback>\s*(.*?)\s*</feedback>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ======================== Helper: build prompt for one index ========================
processor = AutoProcessor.from_pretrained(MODEL_PATH)


def build_prompt(idx):
    """Build a single prompt on-the-fly to avoid holding all in memory."""
    sample = ds[idx]
    question = sample["question"]
    answer = sample["answer"]
    images = sample["images"]
    eval_item = wrong_items[idx]

    student_resp = min(eval_item["responses"], key=len)
    if len(student_resp) < 200:
        student_resp = eval_item["responses"][0]

    if do_truncate and len(student_resp) > 1500:
        student_resp = student_resp[:1500] + "\n... [truncated]"

    feedback_text = FEEDBACK_TEMPLATE.format(
        problem=question.strip(),
        student_response=student_resp,
        gold_answer=answer,
    )

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": feedback_text})

    messages = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, _ = process_vision_info(messages)

    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs

    return (
        {"prompt": prompt, "multi_modal_data": mm_data},
        {"idx": idx, "question": question, "gold_answer": answer},
    )


# ======================== Streaming Inference ========================
print(f"[Shard {args.shard}] Generating {NUM_FEEDBACKS} feedbacks each for {len(pending_indices)} problems (chunk_size={CHUNK_SIZE})...")
total_done = len(completed)

for chunk_start in range(0, len(pending_indices), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(pending_indices))
    chunk_indices = pending_indices[chunk_start:chunk_end]

    # Build prompts only for this chunk
    chunk_prompts = []
    chunk_meta = []
    for idx in chunk_indices:
        prompt, meta = build_prompt(idx)
        chunk_prompts.append(prompt)
        chunk_meta.append(meta)

    try:
        outputs = llm.generate(chunk_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Chunk {chunk_start//CHUNK_SIZE + 1} FAILED: {e}")
        print(f"[Shard {args.shard}] Falling back to one-by-one...")
        for i, (prompt, meta) in enumerate(zip(chunk_prompts, chunk_meta)):
            try:
                single_out = llm.generate([prompt], sampling_params)
                raw_outputs = [o.text for o in single_out[0].outputs]
            except Exception as e2:
                print(f"[Shard {args.shard}]   idx={meta['idx']} SKIPPED: {e2}")
                raw_outputs = ["[GENERATION_ERROR]"] * NUM_FEEDBACKS

            feedbacks = []
            for raw in raw_outputs:
                fb = extract_feedback(raw)
                feedbacks.append(fb if fb else raw)

            result = {
                "idx": meta["idx"],
                "question": meta["question"],
                "gold_answer": meta["gold_answer"],
                "feedbacks": feedbacks,
                "raw_outputs": raw_outputs,
            }
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            total_done += 1

        print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] Chunk {chunk_start//CHUNK_SIZE + 1} done (with fallback)")
        continue

    for i, output in enumerate(outputs):
        raw_outputs = [o.text for o in output.outputs]
        meta = chunk_meta[i]

        feedbacks = []
        for raw in raw_outputs:
            fb = extract_feedback(raw)
            feedbacks.append(fb if fb else raw)

        result = {
            "idx": meta["idx"],
            "question": meta["question"],
            "gold_answer": meta["gold_answer"],
            "feedbacks": feedbacks,
            "raw_outputs": raw_outputs,
        }

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] Chunk {chunk_start//CHUNK_SIZE + 1} done")

print(f"\n[Shard {args.shard}] Done! Results saved to {OUTPUT_PATH}")
