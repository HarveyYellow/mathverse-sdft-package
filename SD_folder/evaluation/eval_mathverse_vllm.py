import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

import argparse
import json
import re
from pathlib import Path
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content, grade_answer

TEST_DATA = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT/test"
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 16

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--processor_path", type=str, default=None)
parser.add_argument("--mode", type=str, default="greedy",
                    choices=["greedy", "sampling"],
                    help="greedy: temperature=0, n=1; sampling: temperature=1, n=8 (reports top1 + top8)")
args = parser.parse_args()

proc_path = args.processor_path or args.model_path
suffix = "_greedy" if args.mode == "greedy" else "_sampling"
out_path = Path(args.output_dir) / f"{args.label}{suffix}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

if out_path.exists():
    print(f"[{args.label}] Already evaluated ({args.mode}), skipping. -> {out_path}")
    exit(0)

def extract_boxed(text):
    """Use mathruler's extract_boxed_content (same as GRPO training reward)."""
    return extract_boxed_content(text)

def match_answer(pred, gt):
    """Use mathruler's grade_answer (same as GRPO training reward)."""
    if not pred:
        return False
    return grade_answer(pred, gt)

print(f"[{args.label}] Loading model: {args.model_path}")
print(f"[{args.label}] Processor: {proc_path}")
print(f"[{args.label}] Mode: {args.mode}")

llm = LLM(
    model=args.model_path,
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 5},
)

if args.mode == "greedy":
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, n=1)
else:
    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=MAX_NEW_TOKENS, n=8)

processor = AutoProcessor.from_pretrained(proc_path)

print(f"[{args.label}] Loading test data: {TEST_DATA}")
ds = load_from_disk(TEST_DATA)
print(f"[{args.label}] {len(ds)} test samples")

SYSTEM_PROMPT = "Solve this problem. Put your final answer in \\boxed{}. If the question provides multiple-choice options, only put the option letter in the box."

print(f"[{args.label}] Building prompts...")
all_prompts = []
all_answers = []
all_questions = []

for i in range(len(ds)):
    sample = ds[i]
    question = sample["question"]
    answer = sample.get("ground_truth", sample["answer"])
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

    all_prompts.append({"prompt": prompt, "multi_modal_data": mm_data})
    all_answers.append(answer)
    all_questions.append(question)

print(f"[{args.label}] Running inference (batch_size={BATCH_SIZE})...")
correct_greedy = 0
correct_top1 = 0
correct_top8 = 0
total = 0
details = []

for start in range(0, len(all_prompts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(all_prompts))
    batch = all_prompts[start:end]

    try:
        outputs = llm.generate(batch, sampling_params)
    except Exception as e:
        print(f"  Batch {start}-{end} failed: {e}, trying one-by-one...")
        outputs = []
        for p in batch:
            try:
                outputs.extend(llm.generate([p], sampling_params))
            except Exception:
                outputs.append(None)

    for i, output in enumerate(outputs):
        idx = start + i
        answer = all_answers[idx]
        question = all_questions[idx]

        if output is None:
            total += 1
            details.append({"idx": idx, "correct": False, "error": "generation_failed"})
            continue

        if args.mode == "greedy":
            gen_text = output.outputs[0].text
            extracted = extract_boxed(gen_text)
            is_correct = match_answer(extracted, answer)
            correct_greedy += int(is_correct)
            total += 1
            details.append({
                "idx": idx,
                "question": question,
                "ground_truth": answer,
                "extracted": extracted,
                "correct": bool(is_correct),
                "output": gen_text,
            })
        else:
            responses = [o.text for o in output.outputs]
            extracted_list = [extract_boxed(r) for r in responses]
            correct_flags = [match_answer(e, answer) for e in extracted_list]
            top1_correct = correct_flags[0]
            top8_correct = any(correct_flags)
            correct_top1 += int(top1_correct)
            correct_top8 += int(top8_correct)
            total += 1
            details.append({
                "idx": idx,
                "question": question,
                "ground_truth": answer,
                "top1_correct": bool(top1_correct),
                "top8_correct": bool(top8_correct),
                "num_correct_in_8": sum(correct_flags),
                "extracted_list": extracted_list,
                "correct_flags": correct_flags,
                "responses": responses,
            })

    if total > 0 and (start // BATCH_SIZE + 1) % 10 == 0:
        if args.mode == "greedy":
            print(f"  [{total}/{len(ds)}] running acc: {correct_greedy/total:.2%}")
        else:
            print(f"  [{total}/{len(ds)}] top1: {correct_top1/total:.2%}, top8: {correct_top8/total:.2%}")

if args.mode == "greedy":
    accuracy = correct_greedy / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"[{args.label}] Greedy Accuracy: {correct_greedy}/{total} = {accuracy:.2%}")
    print(f"{'='*60}\n")
    result = {
        "label": args.label, "mode": "greedy",
        "model_path": args.model_path,
        "accuracy": accuracy, "correct": correct_greedy, "total": total,
        "details": details,
    }
else:
    acc_top1 = correct_top1 / total if total > 0 else 0.0
    acc_top8 = correct_top8 / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"[{args.label}] Sampling Top1: {correct_top1}/{total} = {acc_top1:.2%}")
    print(f"[{args.label}] Sampling Top8: {correct_top8}/{total} = {acc_top8:.2%}")
    print(f"{'='*60}\n")
    result = {
        "label": args.label, "mode": "sampling",
        "model_path": args.model_path,
        "accuracy_top1": acc_top1, "correct_top1": correct_top1,
        "accuracy_top8": acc_top8, "correct_top8": correct_top8,
        "total": total,
        "details": details,
    }

with open(out_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"[{args.label}] Saved -> {out_path}")
