"""Step 2 (v3): Generate contrastive visual feedback per all-wrong problem.

Pipeline:
  1. Model describes image visual elements
  2. Model describes what student assumed
  3. Model finds the ONE contradicting visual element
  4. Extract structured feedback (ELEMENT/CORRECT/WRONG/CHECK)

Uses mathruler re-scoring for consistent all-wrong filtering.

Usage:
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python feedback/step2_gen_feedback_v3_mathverse.py \
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
from vllm import LLM, SamplingParams
from datasets import load_from_disk
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content, grade_answer

# ======================== Config ========================
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
EVAL_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
MATHRULER_INDICES_PATH = os.path.join(BASE, "mathruler_all_wrong_indices.json")
OUTPUT_DIR = os.path.join(BASE, "mathverse_feedback_v3")
MAX_NEW_TOKENS = 1024  # longer to allow Step 1/2/3 reasoning
CHUNK_SIZE = 2
MAX_PROMPT_TOKENS = 6000

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/feedbacks_shard{args.shard}.jsonl"

# ======================== Load mathruler all-wrong ========================
with open(MATHRULER_INDICES_PATH) as f:
    all_wrong_indices = sorted(json.load(f))

# ======================== Load eval results ========================
print(f"[Shard {args.shard}] Loading eval results...")
eval_results = {}
for f in sorted(glob.glob(f"{EVAL_DIR}/eval_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            if item["idx"] not in eval_results:
                eval_results[item["idx"]] = item

# Shard & resume
shard_indices = all_wrong_indices[args.shard::args.num_shards]
completed = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as fh:
        for line in fh:
            completed.add(json.loads(line)["idx"])
    print(f"[Shard {args.shard}] Resuming: {len(completed)} done")

pending_indices = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] Total: {len(all_wrong_indices)}, shard: {len(shard_indices)}, pending: {len(pending_indices)}")

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
    n=1,
    temperature=0.7,
    top_p=0.95,
    max_tokens=MAX_NEW_TOKENS,
)

# ======================== V3 Prompt ========================
FEEDBACK_GEN_TEMPLATE = """A student solved this problem incorrectly. Analyze the image and find what they saw wrong.

## Problem
{problem}

## Correct Answer
{gold_answer}

## Student's Wrong Answer Path
{student_response}

## Task

### Step 1: Describe what you see in the image
List every visual element:
- Figure type (circle, triangle, graph, etc.)
- All labeled points, values, and angles
- Spatial positions (which point is where, what connects to what)

### Step 2: Describe what the student saw
Based on the student's solution, list what visual elements they assumed:
- What figure type did they think it was?
- What positions/values/relationships did they assume?

### Step 3: Find the contradiction
Compare Step 1 and Step 2. Identify the ONE critical visual element where they differ.

Output the contradiction as:

<feedback>
ELEMENT: (the visual element that was misread)
CORRECT: (what the image actually shows)
WRONG: (what the student assumed)
CHECK: (what to look at in the image to verify)
</feedback>

Rules:
- Each field must be under 30 words
- Focus on visual/spatial facts, not math reasoning
- ONE element only — the most critical one"""

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer


def extract_feedback(text):
    """Extract structured feedback from <feedback> tags."""
    m = re.search(r"<feedback>\s*(.*?)\s*</feedback>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def build_prompt(idx):
    sample = ds[idx]
    question = sample["question"]
    answer = sample["answer"]
    images = sample["images"]
    eval_item = eval_results.get(idx)
    if not eval_item:
        return None, None

    # Pick shortest wrong response
    student_resp = min(eval_item["responses"], key=len)
    if len(student_resp) < 200:
        student_resp = eval_item["responses"][0]
    if len(student_resp) > 1500:
        student_resp = student_resp[:1500] + "\n... [truncated]"

    text = FEEDBACK_GEN_TEMPLATE.format(
        problem=question.strip(),
        gold_answer=answer,
        student_response=student_resp,
    )

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
print(f"[Shard {args.shard}] Generating V3 feedback for {len(pending_indices)} problems...")
total_done = len(completed)

for chunk_start in range(0, len(pending_indices), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(pending_indices))
    chunk_indices = pending_indices[chunk_start:chunk_end]

    chunk_prompts = []
    chunk_meta = []
    skipped = []

    for idx in chunk_indices:
        prompt, meta = build_prompt(idx)
        if prompt is None:
            skipped.append(idx)
            continue
        chunk_prompts.append(prompt)
        chunk_meta.append(meta)

    for idx in skipped:
        sample = ds[idx]
        result = {
            "idx": idx, "question": sample["question"], "gold_answer": sample["answer"],
            "feedback": None, "raw_output": None, "skipped": True,
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
        print(f"[Shard {args.shard}] WARNING: output/meta mismatch, skipping chunk")
        continue

    for i, output in enumerate(outputs):
        meta = chunk_meta[i]
        raw = output.outputs[0].text
        extracted = extract_feedback(raw)

        result = {
            "idx": meta["idx"],
            "question": meta["question"],
            "gold_answer": meta["gold_answer"],
            "feedback": extracted,
            "raw_output": raw,
        }
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}]")

print(f"\n[Shard {args.shard}] Done! Results: {OUTPUT_PATH}")
