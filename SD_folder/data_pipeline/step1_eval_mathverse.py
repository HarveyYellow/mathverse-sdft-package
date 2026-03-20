"""Step 1: Evaluate each training question 8 times, record all outputs.

Uses vLLM with data parallel (1 GPU per shard).

Usage:
    # Run 4 shards in parallel (one per GPU):
    for SHARD in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD python step1_eval_mathverse.py --shard $SHARD --num_shards 4 &
    done
    wait
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

import argparse
import json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content, grade_answer

# ======================== Config ========================
MODEL_PATH = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct"
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
OUTPUT_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
NUM_RESPONSES = 8
MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 16

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/eval_shard{args.shard}.jsonl"

# ======================== Answer Check (mathruler — same as GRPO training) ========================
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

# ======================== Load Dataset ========================
print(f"[Shard {args.shard}] Loading dataset...")
ds = load_from_disk(DATASET_PATH + "/train")
total_samples = len(ds)

all_indices = list(range(total_samples))
shard_indices = all_indices[args.shard::args.num_shards]
pending_indices = [i for i in shard_indices if i not in completed]
print(f"[Shard {args.shard}] Total={total_samples}, this shard={len(shard_indices)}, pending={len(pending_indices)}")

if not pending_indices:
    print(f"[Shard {args.shard}] Nothing to do!")
    exit(0)

# ======================== Load vLLM Model ========================
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
    n=NUM_RESPONSES,
    temperature=1.0,
    top_p=0.9,
    max_tokens=MAX_NEW_TOKENS,
)

# ======================== Build Prompts ========================
SYSTEM_PROMPT = "Solve this problem. Put your final answer in \\boxed{}. If the question provides multiple-choice options, only put the option letter in the box."

print(f"[Shard {args.shard}] Building prompts...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

batch_prompts = []
batch_indices = []
batch_answers = []
batch_questions = []

for idx in pending_indices:
    sample = ds[idx]
    question = sample["question"]
    answer = sample["answer"]
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
    img_inputs, _ = process_vision_info(messages)

    mm_data = {}
    if img_inputs:
        mm_data["image"] = img_inputs

    batch_prompts.append({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })
    batch_indices.append(idx)
    batch_answers.append(answer)
    batch_questions.append(question)

# ======================== Batch Inference ========================
print(f"[Shard {args.shard}] Starting inference on {len(batch_prompts)} samples (chunk_size={CHUNK_SIZE})...")
total_done = len(completed)

for chunk_start in range(0, len(batch_prompts), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(batch_prompts))
    chunk_prompts = batch_prompts[chunk_start:chunk_end]

    try:
        outputs = llm.generate(chunk_prompts, sampling_params)
    except Exception as e:
        print(f"[Shard {args.shard}] Batch failed: {e}, trying one-by-one...")
        outputs = []
        for p in chunk_prompts:
            try:
                outputs.extend(llm.generate([p], sampling_params))
            except:
                outputs.append(None)

    for i, output in enumerate(outputs):
        if output is None:
            continue
        gi = chunk_start + i
        idx = batch_indices[gi]
        answer = batch_answers[gi]
        question = batch_questions[gi]

        responses = [o.text for o in output.outputs]
        correct_flags = [check_answer(answer, resp) for resp in responses]

        result = {
            "idx": idx,
            "question": question,
            "answer": answer,
            "top1": correct_flags[0],
            "top8": any(correct_flags),
            "num_correct_in_8": sum(correct_flags),
            "correct_flags": correct_flags,
            "responses": responses,
        }

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        total_done += 1

    print(f"[Shard {args.shard}] [{total_done}/{len(shard_indices)}] Chunk {chunk_start//CHUNK_SIZE + 1} done")

print(f"\n[Shard {args.shard}] Done! Results saved to {OUTPUT_PATH}")
