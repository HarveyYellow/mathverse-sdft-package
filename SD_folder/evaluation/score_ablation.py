#!/usr/bin/env python3
"""Score ablation evaluation results with Top1-Avg metric."""
import json
import glob
import os

from mathruler.grader import extract_boxed_content, grade_answer

RESULT_DIR = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder/eval_results"
PATTERNS = ["sd_ans_only_ablation_*.json"]

def rescore_greedy(details):
    correct = 0
    total = 0
    for d in details:
        gt = d.get("ground_truth", "")
        text = d.get("output", "")
        if not gt:
            continue
        total += 1
        extracted = extract_boxed_content(text)
        if grade_answer(extracted, gt):
            correct += 1
    return correct, total

def rescore_sampling(details):
    total = 0
    sum_avg = 0.0
    top1_correct = 0
    top8_correct = 0
    for d in details:
        gt = d.get("ground_truth", "")
        responses = d.get("responses", [])
        if not gt or not responses:
            continue
        total += 1
        flags = []
        for r in responses:
            extracted = extract_boxed_content(r)
            flags.append(grade_answer(extracted, gt))
        n_correct = sum(flags)
        sum_avg += n_correct / len(flags)
        top1_correct += int(flags[0])
        top8_correct += int(any(flags))
    return total, sum_avg, top1_correct, top8_correct

print("=" * 70)
print("  MathVerse Ablation (ans_only) — Re-scored with mathruler.grader")
print("=" * 70)

files = sorted(glob.glob(os.path.join(RESULT_DIR, "sd_ans_only_ablation_*.json")))
for f in files:
    label = os.path.basename(f).replace(".json", "")
    with open(f) as fp:
        data = json.load(fp)
    mode = data.get("mode", "")
    details = data.get("details", [])

    if mode == "greedy":
        correct, total = rescore_greedy(details)
        acc = correct / total if total > 0 else 0
        print(f"\n  {label}")
        print(f"    Greedy:    {correct}/{total} = {acc:.2%}")
    else:
        total, sum_avg, top1_c, top8_c = rescore_sampling(details)
        if total > 0:
            top1_avg = sum_avg / total
            top1_acc = top1_c / total
            top8_acc = top8_c / total
        else:
            top1_avg = top1_acc = top8_acc = 0
        print(f"\n  {label}")
        print(f"    Top1-Orig: {top1_c}/{total} = {top1_acc:.2%}")
        print(f"    Top1-Avg:  {top1_avg:.2%}")
        print(f"    Top8:      {top8_c}/{total} = {top8_acc:.2%}")

print(f"\n{'=' * 70}")
