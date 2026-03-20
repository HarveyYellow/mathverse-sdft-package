"""Step 3: Build training datasets for mathverse self-distillation.

Creates TWO dataset variants:
  - ref_only:  all-wrong problems get reflection only (no answer)
  - ref_ans:   all-wrong problems get reflection + correct answer

Supports two reflection sources via --reflection_dir:
  - mathverse_reflection_n8_truncated  (student response was truncated)
  - mathverse_reflection_n8_full       (student response was kept in full)

Usage:
    # Truncated version:
    python step3_build_mathverse_train.py --reflection_dir mathverse_reflection_n8_truncated --suffix _trunc

    # Full version:
    python step3_build_mathverse_train.py --reflection_dir mathverse_reflection_n8_full --suffix _full
"""

import argparse
import json
import glob
import os
from datasets import load_from_disk, Dataset

# ======================== Args ========================
parser = argparse.ArgumentParser()
parser.add_argument("--reflection_dir", type=str, required=True,
                    help="Directory with reflection_shard*.jsonl files")
parser.add_argument("--suffix", type=str, default="",
                    help="Suffix for output directory names (e.g., _trunc or _full)")
args = parser.parse_args()

# ======================== Paths ========================
BASE = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder"
DATASET_PATH = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT"
EVAL_DIR = os.path.join(BASE, "mathverse_eval_results_vllm")
REFLECTION_DIR = os.path.join(BASE, args.reflection_dir)
OUTPUT_REF_ONLY = os.path.join(BASE, f"data/mathverse_sdft_ref_only_{args.suffix}/train")
OUTPUT_REF_ANS = os.path.join(BASE, f"data/mathverse_sdft_ref_ans_{args.suffix}/train")

print(f"Reflection dir: {REFLECTION_DIR}")
print(f"Output ref_only: {OUTPUT_REF_ONLY}")
print(f"Output ref_ans:  {OUTPUT_REF_ANS}")

# ======================== Load eval results ========================
print("\nLoading eval results...")
eval_results = {}
for f in sorted(glob.glob(f"{EVAL_DIR}/eval_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            if item["idx"] not in eval_results:
                eval_results[item["idx"]] = item
print(f"  Loaded {len(eval_results)} eval results")

# ======================== Load reflections ========================
print("Loading reflections (n=8)...")
reflections = {}
for f in sorted(glob.glob(f"{REFLECTION_DIR}/reflection_shard*.jsonl")):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            if item["idx"] not in reflections:
                reflections[item["idx"]] = item
print(f"  Loaded reflections for {len(reflections)} problems")

# ======================== Load dataset ========================
print("Loading mathverse dataset...")
ds = load_from_disk(DATASET_PATH + "/train")
print(f"  {len(ds)} samples")

# ======================== Build training data ========================
print("\nBuilding training data...")
records_ref_only = []
records_ref_ans = []
stats = {"correct_cot": 0, "reflection": 0, "answer_only": 0, "missing_reflection": 0}

for idx in range(len(ds)):
    sample = ds[idx]
    eval_item = eval_results.get(idx)
    if eval_item is None:
        print(f"  WARNING: idx={idx} missing from eval results, skipping")
        continue

    base = {
        "images": sample["images"],
        "question": sample["question"],
        "answer": sample["answer"],
    }

    if eval_item["num_correct_in_8"] > 0:
        # Has correct responses — pick the shortest correct CoT
        correct_responses = [
            resp for resp, flag in zip(eval_item["responses"], eval_item["correct_flags"])
            if flag
        ]
        shortest_correct = min(correct_responses, key=len)
        record = {**base, "teacher_type": "correct_cot", "teacher_extra": shortest_correct}
        records_ref_only.append(record)
        records_ref_ans.append(record)
        stats["correct_cot"] += 1
    else:
        # All wrong — check reflections
        ref_item = reflections.get(idx)
        if ref_item is None:
            record = {**base, "teacher_type": "answer_only", "teacher_extra": ""}
            records_ref_only.append(record)
            records_ref_ans.append(record)
            stats["answer_only"] += 1
            stats["missing_reflection"] += 1
        else:
            candidates = [
                r for r in ref_item["reflections"]
                if len(r.split()) <= 150 and "[GENERATION_ERROR]" not in r
            ]
            if candidates:
                shortest_ref = min(candidates, key=lambda r: len(r.split()))
                # reflection_only version
                records_ref_only.append({
                    **base, "teacher_type": "reflection", "teacher_extra": shortest_ref,
                })
                # reflection_with_answer version
                records_ref_ans.append({
                    **base, "teacher_type": "reflection_with_answer", "teacher_extra": shortest_ref,
                })
                stats["reflection"] += 1
            else:
                record = {**base, "teacher_type": "answer_only", "teacher_extra": ""}
                records_ref_only.append(record)
                records_ref_ans.append(record)
                stats["answer_only"] += 1

print(f"\nDataset Statistics:")
print(f"  Total:            {len(records_ref_only)}")
print(f"  correct_cot:      {stats['correct_cot']}")
print(f"  reflection:       {stats['reflection']}")
print(f"  answer_only:      {stats['answer_only']} (of which {stats['missing_reflection']} missing reflection data)")

# ======================== Save ========================
for name, records, out_dir in [
    ("ref_only", records_ref_only, OUTPUT_REF_ONLY),
    ("ref_ans", records_ref_ans, OUTPUT_REF_ANS),
]:
    print(f"\nSaving {name} to {out_dir}...")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    train_ds = Dataset.from_list(records)
    train_ds.save_to_disk(out_dir)
    print(f"  Saved {len(train_ds)} examples.")

    # Verify
    verify = load_from_disk(out_dir)
    from collections import Counter
    type_counts = Counter(verify["teacher_type"])
    print(f"  Columns: {verify.column_names}")
    print(f"  Teacher type distribution:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

print("\nDone!")
