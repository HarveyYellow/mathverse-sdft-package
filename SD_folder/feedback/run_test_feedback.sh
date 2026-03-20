#!/bin/bash
# Test whether feedback improves model performance on all-wrong problems
# 4 shards, 1 GPU each
set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
cd "$BASE"

export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

echo "[$(date)] Starting feedback test (4 shards)..."

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python feedback/step3_test_feedback_mathverse.py \
        --shard $SHARD --num_shards 4 &
    sleep 60
done
wait

echo "[$(date)] Done!"
echo "Results:"
wc -l mathverse_feedback_test/test_shard*.jsonl

# Print summary
python3 -c "
import json, glob

all_items = []
for f in sorted(glob.glob('mathverse_feedback_test/test_shard*.jsonl')):
    with open(f) as fh:
        for line in fh:
            all_items.append(json.loads(line))

total = len(all_items)
if total == 0:
    print('No results found!')
    exit()

top1 = sum(1 for r in all_items if r['top1'])
top8 = sum(1 for r in all_items if r['top8'])
avg_correct = sum(r['num_correct_in_8'] for r in all_items) / total

print(f'')
print(f'=== Feedback Test Results (all-wrong problems) ===')
print(f'Total problems tested: {total}')
print(f'Top-1 accuracy (with feedback): {top1}/{total} = {top1*100/total:.1f}%')
print(f'Top-8 accuracy (with feedback): {top8}/{total} = {top8*100/total:.1f}%')
print(f'Avg correct in 8 attempts:      {avg_correct:.2f}')
print(f'')
print(f'Baseline (without feedback): 0/8 correct for all {total} problems')
print(f'Improvement: {top8} problems now solvable with feedback ({top8*100/total:.1f}%)')
"
