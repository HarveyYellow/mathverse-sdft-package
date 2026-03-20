"""Re-score all eval results using mathruler.grader (same as GRPO training)."""
import json, glob, os

from mathruler.grader import extract_boxed_content, grade_answer

RESULTS_DIR = "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder/eval_results"

# Only re-score epoch-level results
targets = [
    "base_qwen3vl8b",
    "grpo_epoch1_step17", "grpo_epoch2_step34", "grpo_epoch3_step51", "grpo_epoch4_step68",
    "sd_ref_only_trunc_ep1", "sd_ref_only_trunc_ep2",
    "sd_ref_ans_trunc_ep1", "sd_ref_ans_trunc_ep2",
    "sd_ref_only_full_ep1", "sd_ref_only_full_ep2",
    "sd_ref_ans_full_ep1", "sd_ref_ans_full_ep2",
]

summary = {}

for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    with open(fpath) as f:
        data = json.load(f)

    label = data.get("label", "")
    mode = data.get("mode", "")

    if label not in targets:
        continue

    details = data.get("details", [])
    if not details:
        continue

    if label not in summary:
        summary[label] = {}

    total = 0

    if mode == "sampling":
        correct_top1 = 0
        correct_top8 = 0
        old_top1 = data.get("correct_top1", 0)
        old_top8 = data.get("correct_top8", 0)

        for d in details:
            responses = d.get("responses", [])
            gt = str(d.get("ground_truth", ""))
            if not responses:
                continue
            total += 1

            extracted_list = [extract_boxed_content(r) for r in responses]
            correct_flags = [grade_answer(e, gt) if e else False for e in extracted_list]

            if correct_flags[0]:
                correct_top1 += 1
            if any(correct_flags):
                correct_top8 += 1

        acc_top1 = correct_top1 / total if total > 0 else 0
        acc_top8 = correct_top8 / total if total > 0 else 0
        summary[label]["top1"] = acc_top1
        summary[label]["top8"] = acc_top8
        summary[label]["total"] = total
        print(f"[{label}] sampling: top1={correct_top1}/{total}={acc_top1:.2%} (was {old_top1}), top8={correct_top8}/{total}={acc_top8:.2%} (was {old_top8})")

    else:  # greedy
        correct = 0
        old_correct = data.get("correct", 0)

        for d in details:
            output = d.get("output", "")
            gt = str(d.get("ground_truth", ""))
            if not output and output != "":
                continue
            total += 1

            extracted = extract_boxed_content(output)
            is_correct = grade_answer(extracted, gt) if extracted else False

            if is_correct:
                correct += 1

        acc = correct / total if total > 0 else 0
        summary[label]["greedy"] = acc
        summary[label]["total"] = total
        print(f"[{label}] greedy: {correct}/{total}={acc:.2%} (was {old_correct}/{total})")

# Print summary table
print()
print(f"{'Model':<35} {'Greedy':>8} {'Top1(t=1)':>10} {'Top8(t=1)':>10}")
print("-" * 65)
for label in targets:
    if label not in summary:
        print(f"{label:<35} {'(missing)':>8}")
        continue
    m = summary[label]
    g = f"{m['greedy']:.2%}" if 'greedy' in m else "  -"
    t1 = f"{m['top1']:.2%}" if 'top1' in m else "  -"
    t8 = f"{m['top8']:.2%}" if 'top8' in m else "  -"
    print(f"{label:<35} {g:>8} {t1:>10} {t8:>10}")
