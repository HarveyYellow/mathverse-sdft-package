# MathVerse Self-Distillation with Adaptive Feedback

Self-distillation framework for visual math reasoning on the MathVerse dataset using Qwen3-VL-8B-Instruct. The system generates per-problem adaptive teacher prompts (correct CoT / error reflection / answer-only) and supports online feedback injection during training.

## Data Notice

> **The `.arrow` data files in this repository have been truncated to 50 samples each** (from the original ~2000+ samples) to keep the repository lightweight and enable quick case study inspection. These samples are sufficient for understanding the data format and schema, but are **not** suitable for training or full evaluation. To reproduce the full dataset, re-run the [data pipeline](#data-pipeline-3-step).

## Table of Contents

- [Overview](#overview)
- [Pipeline Overview](#pipeline-overview)
- [Directory Structure](#directory-structure)
- [Environment Setup](#environment-setup)
- [Data Pipeline (3-Step)](#data-pipeline-3-step)
- [Training](#training)
- [Evaluation](#evaluation)
- [Key Hyperparameters](#key-hyperparameters)
- [File Descriptions](#file-descriptions)

---

## Overview

### Core Idea

Standard knowledge distillation uses the same teacher prompt for all problems. This framework instead **adapts the teacher prompt per-problem** based on the base model's actual performance:

| Base Model Performance | Teacher Type | Teacher Prompt Contains |
|------------------------|--------------|------------------------|
| At least 1/8 correct | `correct_cot` | Correct answer + shortest correct CoT as reference |
| All 8 wrong | `reflection_with_answer` | Correct answer + error reflection from failed attempts |
| All 8 wrong (no reflection) | `answer_only` | Correct answer only |

### Online Feedback Mechanism

During training, `FeedbackDistilTrainer` further adapts in real-time:

1. The student model generates a response to the problem
2. The response is checked against the ground truth via `\boxed{}` extraction
3. **If the student answers incorrectly**, the teacher prompt is augmented with:
   ```
   [Student's Previous Attempt]: The student answered "{wrong_answer}" but this is incorrect.
   Considering the student's mistake, construct a solution that clearly avoids this error.
   ```
4. The teacher then generates a completion conditioned on this augmented prompt
5. The student is trained to match the teacher's (now more targeted) output via KL divergence

This creates a closed-loop where the teacher's guidance is dynamically tailored to the student's specific mistakes at each training step.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Evaluate base model on each training question (×8)         │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Input:  MathVerse train set (2283 questions + images)   │      │
│  │  Model:  Qwen3-VL-8B-Instruct (base, unmodified)        │      │
│  │  Method: 8× sampling (temperature=1.0, top_p=0.9)       │      │
│  │  Output: Per-question eval results (correct/wrong flags) │      │
│  └──────────────────────┬───────────────────────────────────┘      │
│                         │                                           │
│                         ▼                                           │
│       ┌─────────────────┴─────────────────┐                        │
│       │   ≥1/8 correct    │   0/8 correct  │                       │
│       │   (1202 problems) │  (1043 problems)│                      │
│       └────────┬──────────┴───────┬────────┘                       │
│                │                  │                                  │
│                │                  ▼                                  │
│                │    Step 2: Generate reflections for all-wrong       │
│                │    ┌────────────────────────────────────────┐      │
│                │    │  Prompt: "Review student's wrong        │      │
│                │    │    solution, identify error in 3 lines" │      │
│                │    │  8 reflections per problem              │      │
│                │    │  Pick shortest valid (≤150 words)       │      │
│                │    └──────────────────┬─────────────────────┘      │
│                │                       │                             │
│                ▼                       ▼                             │
│  Step 3: Assemble training dataset                                  │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  correct_cot (1202):         answer + shortest correct CoT│     │
│  │  reflection_with_answer (1043): answer + error reflection │     │
│  │  Output: HuggingFace Dataset (images, question, answer,   │     │
│  │          teacher_type, teacher_extra)                      │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          TRAINING                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Student Model ◄──── KL Loss ────► Teacher Model          │     │
│  │  (trainable)          ▲             (frozen, EMA-synced)  │     │
│  │       │               │                    │              │     │
│  │       ▼               │                    ▼              │     │
│  │  Student Prompt   Online Feedback     Teacher Prompt      │     │
│  │  (plain question)  Injection          (answer + reflection│     │
│  │                   (if student wrong,   + optional feedback)│     │
│  │                    inject error info)                      │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         EVALUATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Greedy:   temperature=0, n=1   → Greedy Accuracy         │     │
│  │  Sampling: temperature=1, n=8   → Top1-Avg / Top8         │     │
│  │  Grader:   mathruler.grader (same as GRPO reward)         │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
mathverse_sdft_package/
│
├── README.md                        # English documentation
├── README_CN.md                     # Chinese documentation
│
├── core/                            # Core training framework
│   ├── distil_trainer.py            # DistilTrainer — base trainer class (extends TRL)
│   └── distil_config.py             # DistilConfig — training hyperparameter definition
│
├── data/                            # All datasets
│   ├── raw/                         # Unprocessed original data
│   │   └── sdft_mathverse/
│   │       ├── train/               # 2283 training samples (images + question + answer)
│   │       └── test/                # 257 test samples
│   │
│   ├── intermediate/                # Pipeline intermediate outputs
│   │   ├── mathverse_eval_results_vllm/        # Step 1 output: 8× sampling eval per question
│   │   ├── mathverse_reflection_n8_truncated/  # Step 2 output: reflections (truncated)
│   │   └── mathverse_reflection_n8_full/       # Step 2 output: reflections (full)
│   │
│   └── processed/                   # Final training datasets (Step 3 output)
│       ├── mathverse_sdft_ref_ans_trunc/    # reflection_with_answer + correct_cot (truncated)
│       ├── mathverse_sdft_ref_only_trunc/   # reflection only, no answer (truncated)
│       ├── mathverse_sdft_ref_ans_full/     # reflection_with_answer + correct_cot (full)
│       ├── mathverse_sdft_ref_only_full/    # reflection only, no answer (full)
│       └── mathverse_sdft_ans_only_ablation/  # ablation: all reflections → answer_only
│
├── data_pipeline/                   # 3-step data generation pipeline
│   ├── step1_eval_mathverse.py      # Step 1: evaluate base model (8× sampling per question)
│   ├── step2_gen_reflection_mathverse.py  # Step 2: generate error reflections for all-wrong problems
│   └── step3_build_mathverse_train.py     # Step 3: assemble final training dataset
│
├── training/                        # Training entry scripts
│   ├── main_vlm_mathverse_qwen3.py              # WITH online feedback (FeedbackDistilTrainer)
│   └── main_vlm_mathverse_qwen3_no_feedback.py  # WITHOUT online feedback (DistilTrainer, for ablation)
│
├── evaluation/                      # Evaluation scripts
│   ├── eval_mathverse_vllm.py       # Main evaluation: greedy + 8× sampling via vLLM
│   ├── rescore_all.py               # Re-score existing results with mathruler.grader
│   └── score_ablation.py            # Compute Top1-Avg metric for ablation experiments
│
├── eval_results/                    # Evaluation output JSONs + summaries
│   ├── mathverse/                   # SD + GRPO per-step eval results (83 files)
│   │   ├── base_qwen3vl8b_*.json   #   base model (greedy + sampling)
│   │   ├── sd_ref_ans_trunc_*.json  #   SD ref_ans_trunc ep1/ep2
│   │   ├── sd_ref_only_trunc_*.json #   SD ref_only_trunc ep1/ep2
│   │   ├── sd_ref_ans_full_*.json   #   SD ref_ans_full ep1/ep2
│   │   ├── sd_ref_only_full_*.json  #   SD ref_only_full ep1/ep2
│   │   ├── sd_ans_only_ablation_*.json  # ablation (no feedback) ep1/ep2
│   │   └── grpo_*.json              #   GRPO per-step (step 1-31) + per-epoch
│   ├── mathverse_4plus4/            # GRPO 4+4 epoch eval results (16 files)
│   ├── eval_results_with_avg.txt    # Final summary (Top1-Avg + Top8 + Greedy)
│   ├── eval_results_all_summary.txt # Earlier summary
│   └── eval_results_rescored.txt    # Re-scored results
│
├── config/                          # Configuration files
│   ├── accelerate_config_gqa.yaml   # Distributed training config (DeepSpeed ZeRO-2, 4 GPU)
│   └── requirements.txt             # Python dependencies
│
└── scripts/                         # Launch scripts
    ├── run_mathverse_data_pipeline.sh    # One-click: run full data pipeline (step1→2→3)
    ├── run_all_mathverse_training.sh     # Full training pipeline (multiple SD variants)
    ├── run_mathverse_no_feedback.sh      # Train without online feedback (ablation)
    ├── run_ablation_ans_only.sh          # Ablation: feedback vs no-feedback
    ├── run_eval_ablation.sh              # Evaluate ablation checkpoints
    └── run_eval_epoch_only.sh            # Batch evaluate all epoch-level checkpoints
```

---

## Environment Setup

### Hardware Requirements

- 4× NVIDIA H100 (or A100) GPUs, 80GB each
- ~60GB CPU RAM for DeepSpeed optimizer offload

### Software Requirements

```bash
pip install -r config/requirements.txt
```

Key dependencies:
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.0 | Core framework |
| `transformers` | 4.57.1 | Model loading, tokenizer |
| `trl` | 0.24.0 | Base trainer class (GRPO/distillation) |
| `vllm` | 0.12.0 | Fast inference for data pipeline + evaluation |
| `accelerate` | 1.11.0 | Distributed training |
| `deepspeed` | 0.18.4 | ZeRO-2 memory optimization |
| `mathruler` | (pip) | Answer extraction & grading (same as GRPO reward) |
| `qwen_vl_utils` | (pip) | Qwen-VL image processing |
| `flashinfer-python` | 0.5.3 | Flash attention for vLLM |

### Model

Download Qwen3-VL-8B-Instruct to local storage:
```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct
```

---

## Data Pipeline (3-Step)

### Prerequisites

MathVerse dataset must be saved as a HuggingFace Dataset at `data/sdft_mathverse/train/` with columns:
- `images`: list of PIL images
- `question`: problem text (may include choice options)
- `answer`: ground truth answer string

### Step 1: Evaluate Base Model

Evaluate each training question 8 times with the unmodified base model to categorize problem difficulty.

```bash
# Run 4 shards in parallel (one per GPU):
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step1_eval_mathverse.py \
        --shard $SHARD --num_shards 4 &
done
wait
```

**Configuration:**
- Sampling: `temperature=1.0, top_p=0.9, n=8, max_tokens=2048`
- Answer matching: `extract_boxed()` → `match_mathverse()` (handles MCQ option mapping + numeric tolerance)
- Output: `mathverse_eval_results_vllm/eval_shard{0,1,2,3}.jsonl`

**Output format** (per question):
```json
{
    "idx": 42,
    "question": "...",
    "answer": "B",
    "responses": ["resp1", "resp2", ..., "resp8"],
    "correct_flags": [true, false, false, true, ...],
    "num_correct_in_8": 2
}
```

### Step 2: Generate Error Reflections

For all-wrong problems (0/8 correct), generate 8 structured error reflections per problem.

```bash
# Truncated version (default, student response ≤ 1500 chars):
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step2_gen_reflection_mathverse.py \
        --shard $SHARD --num_shards 4 --truncate &
done
wait
```

**Reflection prompt template:**
```
You are a math teacher reviewing a student's wrong solution.
Be brief and precise — your entire response must be under 150 words.

## Problem
{problem}

## Student's Solution (WRONG)
{student_response}

## Correct Answer
{gold_answer}

## Task
Identify the specific error in 3 lines:
1. **Error Type** (one of: Image Misreading / Theorem Misapplication /
   Calculation Error / Reasoning Loop / Variable Misassignment /
   Diagram Relationship Error / Option Mapping Error)
2. **Error Detail**: One sentence pinpointing exactly where the reasoning went wrong.
3. **Correction Hint**: One sentence telling the student how to fix it.
```

**Configuration:**
- Sampling: `temperature=0.7, top_p=0.95, frequency_penalty=0.3, n=8, max_tokens=512`
- Output: `mathverse_reflection_n8_truncated/reflection_shard{0,1,2,3}.jsonl`

### Step 3: Assemble Training Dataset

Combine eval results and reflections into the final training dataset.

```bash
python data_pipeline/step3_build_mathverse_train.py \
    --reflection_dir mathverse_reflection_n8_truncated \
    --suffix trunc
```

**Assignment logic:**

| Condition | teacher_type | teacher_extra |
|-----------|-------------|---------------|
| ≥1/8 correct | `correct_cot` | Shortest correct CoT from Step 1 |
| 0/8 correct, has valid reflection | `reflection_with_answer` | Shortest reflection (≤150 words) |
| 0/8 correct, no valid reflection | `answer_only` | empty string |

**Output:** HuggingFace Dataset at `data/mathverse_sdft_ref_ans_trunc/train/` with columns:
- `images`, `question`, `answer`, `teacher_type`, `teacher_extra`

**Typical distribution:** ~1202 correct_cot + ~1043 reflection_with_answer = ~2245 total

---

## Training

### Teacher Prompt Construction

At training time, each sample's teacher prompt is constructed based on `teacher_type`:

```python
# correct_cot: provide answer + reference CoT
"[Correct Answer]: {answer}"
"[Reference Reasoning]: {teacher_extra}"
"Given the correct answer above, construct a clear step-by-step solution..."

# reflection_with_answer: provide answer + error analysis
"[Correct Answer]: {answer}"
"[Error Analysis from a previous attempt]: {teacher_extra}"
"Given the correct answer and error analysis above, construct a clear step-by-step solution..."

# answer_only: provide answer only
"[Correct Answer]: {answer}"
"Given the correct answer above, construct a clear step-by-step solution..."
```

The student prompt is always the **plain question text** without any hints.

### With Online Feedback (Default)

```bash
cd /path/to/mathverse_sdft_package

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3.py \
    --model_name /path/to/Qwen3-VL-8B-Instruct \
    --data_dir data/mathverse_sdft_ref_ans_trunc \
    --output_dir outputs/mathverse_ref_ans_trunc \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none
```

Uses `FeedbackDistilTrainer`: if the student's generated answer is wrong at a training step, the teacher prompt is dynamically augmented with the student's wrong answer before the teacher generates.

### Without Online Feedback (Ablation)

```bash
accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3_no_feedback.py \
    --model_name /path/to/Qwen3-VL-8B-Instruct \
    --data_dir data/mathverse_sdft_ans_only_ablation \
    --output_dir outputs/mathverse_ans_only_ablation \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none
```

Uses base `DistilTrainer`: no online feedback injection. For ablation, the data should also have reflections removed (all converted to `answer_only`).

### Training Architecture

```
Student Model (trainable)
    ├── Generates completion given plain question
    ├── Loss = KL(student_logits || teacher_logits) on completion tokens
    └── Skips first 3 tokens of loss (num_loss_tokens_to_skip=3)

Teacher/Reference Model (frozen, EMA-synced)
    ├── Generates completion given enriched teacher prompt
    ├── EMA sync: ref_model ← α * student + (1-α) * ref_model every step
    └── α = ref_model_mixup_alpha = 0.01
```

---

## Evaluation

### Run Evaluation

```bash
# Greedy (temperature=0, n=1):
python evaluation/eval_mathverse_vllm.py \
    --model_path outputs/mathverse_ref_ans_trunc/checkpoint-35 \
    --processor_path /path/to/Qwen3-VL-8B-Instruct \
    --label sd_ref_ans_trunc_ep1 \
    --output_dir eval_results \
    --mode greedy

# Sampling (temperature=1, n=8):
python evaluation/eval_mathverse_vllm.py \
    --model_path outputs/mathverse_ref_ans_trunc/checkpoint-35 \
    --processor_path /path/to/Qwen3-VL-8B-Instruct \
    --label sd_ref_ans_trunc_ep1 \
    --output_dir eval_results \
    --mode sampling
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Greedy** | Single-draw accuracy with `temperature=0` |
| **Top1-Avg** | `mean(num_correct / 8)` per question — average pass rate across 8 samples |
| **Top8** | At least 1 of 8 samples correct — upper bound of model capability |

All answer matching uses `mathruler.grader.extract_boxed_content` + `grade_answer`, consistent with GRPO training reward.

### Re-score Results

```bash
# Re-score all results with mathruler.grader:
python evaluation/rescore_all.py

# Score ablation experiments with Top1-Avg:
python evaluation/score_ablation.py
```

---

## Key Hyperparameters

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-6 | Learning rate |
| `lr_scheduler_type` | cosine | LR schedule |
| `warmup_ratio` | 0.0 | No warmup |
| `num_train_epochs` | 2 | Training epochs |
| `per_device_train_batch_size` | 1 | Per-GPU batch size |
| `global_batch_size` | 64 | Effective batch (via gradient accumulation) |
| `max_prompt_length` | 8192 | Max input tokens |
| `max_completion_length` | 2048 | Max generation tokens |
| `temperature` | 1.0 | Student generation temperature |
| `top_p` | 0.99 | Student generation top-p |
| `num_generations` | 1 | Completions per prompt per step |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `ref_model_mixup_alpha` | 0.01 | EMA sync rate for teacher model |
| `ref_model_sync_steps` | 1 | Sync every N steps |
| `num_loss_tokens_to_skip` | 3 | Skip first N completion tokens in loss |
| `gradient_checkpointing` | True | Memory optimization |
| `bf16` | True | Mixed precision |

### Distributed Training (DeepSpeed ZeRO-2)

| Parameter | Value |
|-----------|-------|
| `zero_stage` | 2 |
| `offload_optimizer_device` | cpu |
| `offload_param_device` | none |
| `num_processes` | 4 |
| `mixed_precision` | bf16 |

### Data Pipeline

| Step | Key Parameters |
|------|---------------|
| Step 1 (eval) | `temperature=1.0, top_p=0.9, n=8, max_tokens=2048` |
| Step 2 (reflection) | `temperature=0.7, top_p=0.95, frequency_penalty=0.3, n=8, max_tokens=512` |
| Step 3 (assembly) | Shortest correct CoT; shortest reflection ≤150 words |

---

## File Descriptions

### `core/distil_trainer.py`

Extended from TRL's `BaseTrainer`. Implements the self-distillation training loop:
- Student generates completions from plain prompts
- Teacher generates completions from enriched prompts (with answer/reflection)
- KL divergence loss between student and teacher logits
- EMA sync of teacher weights toward student
- Multi-turn generation support for VLM (vision-language model) inputs

### `core/distil_config.py`

Extends `TrainingArguments` with distillation-specific fields:
- `num_generations`, `temperature`, `top_p` for generation
- `max_prompt_length`, `max_completion_length` for sequence lengths
- `sync_ref_model`, `ref_model_sync_steps`, `ref_model_mixup_alpha` for EMA
- `num_loss_tokens_to_skip` to ignore initial tokens in loss computation

### `training/main_vlm_mathverse_qwen3.py`

Training entry with `FeedbackDistilTrainer` (extends `DistilTrainer`):
- `_generate_and_score_completions()`: after student generates, checks answer against ground truth; if wrong, injects `[Student's Previous Attempt]` into teacher prompt
- `_extract_boxed()`: regex extraction of `\boxed{}` content
- `_answers_match()`: string + numeric tolerance matching
- `_generate_single_turn()`: OOM-safe generation with fallback to `max_new_tokens=1`
- Logs `feedback/wrong_ratio` metric per batch

### `training/main_vlm_mathverse_qwen3_no_feedback.py`

Same as above but uses base `DistilTrainer` directly. No `FeedbackDistilTrainer` class, no online feedback injection. For ablation experiments to isolate the effect of feedback.

### `evaluation/eval_mathverse_vllm.py`

Evaluation using vLLM for fast batched inference:
- Supports `greedy` (temperature=0, n=1) and `sampling` (temperature=1, n=8) modes
- Uses `mathruler.grader` for answer extraction and matching (same as GRPO reward)
- Saves full model outputs for post-hoc analysis
- Skip logic: if output file exists, skips re-evaluation
- vLLM config: `tensor_parallel_size=4, gpu_memory_utilization=0.85, max_model_len=8192`
