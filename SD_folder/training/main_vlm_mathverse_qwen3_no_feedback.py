#!/usr/bin/env python3
"""
Self-Distillation on MathVerse dataset with Qwen3-VL-8B — NO online feedback.

Same as main_vlm_mathverse_qwen3.py but uses base DistilTrainer
(no FeedbackDistilTrainer, no online feedback injection).

Usage:
    accelerate launch --config_file accelerate_config_gqa.yaml \
        main_vlm_mathverse_qwen3_no_feedback.py \
        --model_name /home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct \
        --data_dir data/mathverse_sdft_ans_only_ablation \
        --output_dir outputs/mathverse_ans_only_ablation \
        --num_train_epochs 2 --learning_rate 1e-6 \
        --per_device_batch_size 1 --global_batch_size 64 \
        --report_to none
"""

import argparse

import torch
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainerCallback

from distil_trainer import DistilTrainer
from distil_config import DistilConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

TEACHER_SUFFIX = (
    "Think step by step, then put your final answer in \\boxed{your answer}. "
    "If the question provides multiple-choice options, only put the option letter in the box."
)

STUDENT_SUFFIX = ""  # No extra instructions — match benchmark eval prompts


def build_teacher_text(question, answer, teacher_type, teacher_extra):
    """Build teacher prompt text based on per-problem teacher_type."""
    parts = [question.strip()]

    if teacher_type == "correct_cot":
        parts.append(f"\n\n[Correct Answer]: {answer}")
        parts.append(f"\n\n[Reference Reasoning]: {teacher_extra}")
        parts.append(
            "\n\nGiven the correct answer above, construct a clear step-by-step solution "
            "leading to this answer. The reference reasoning is provided for your reference "
            "but may contain errors — verify it against the image before using. "
            "Do not reveal that you were provided with the correct answer or any additional information. "
            + TEACHER_SUFFIX
        )
    elif teacher_type == "reflection":
        parts.append(f"\n\n[Error Analysis from a previous attempt]:\n{teacher_extra}")
        parts.append(
            "\n\nCarefully consider the above error analysis to avoid making the same mistake. "
            + TEACHER_SUFFIX
        )
    elif teacher_type == "reflection_with_answer":
        parts.append(f"\n\n[Correct Answer]: {answer}")
        parts.append(f"\n\n[Error Analysis from a previous attempt]:\n{teacher_extra}")
        parts.append(
            "\n\nGiven the correct answer and error analysis above, construct a clear step-by-step solution. "
            "Do not reveal that you were provided with the correct answer or any additional information. "
            + TEACHER_SUFFIX
        )
    else:  # answer_only
        parts.append(f"\n\n[Correct Answer]: {answer}")
        parts.append(
            "\n\nGiven the correct answer above, construct a clear step-by-step solution "
            "leading to this answer. "
            "Do not reveal that you were provided with the correct answer or any additional information. "
            + TEACHER_SUFFIX
        )

    return "".join(parts)


def build_prompts(example):
    question = example["question"].strip()
    answer = example["answer"].strip()
    teacher_type = example["teacher_type"]
    teacher_extra = example.get("teacher_extra", "")

    student_text = question + STUDENT_SUFFIX
    teacher_text = build_teacher_text(question, answer, teacher_type, teacher_extra)

    student_prompt = [{"role": "user", "content": [
        {"type": "image", "text": None},
        {"type": "text", "text": student_text},
    ]}]

    teacher_prompt = [{"role": "user", "content": [
        {"type": "image", "text": None},
        {"type": "text", "text": teacher_text},
    ]}]

    return {
        "prompt": student_prompt,
        "teacher_prompt": teacher_prompt,
        "ground_truth": answer,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Self-Distillation on MathVerse — NO online feedback")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str,
                    default="/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct")
    p.add_argument("--data_dir", type=str, required=True,
                    help="Path to dataset dir (e.g., data/mathverse_sdft_ans_only_ablation)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--report_to", type=str, default="none", choices=["wandb", "none"])
    p.add_argument("--freeze_vision", action="store_true", default=False)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def freeze_vision_parameters(model):
    total, trainable, frozen = 0, 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "visual" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()
    print("=" * 60)
    print("Parameter Freezing Summary")
    print("=" * 60)
    print(f"  Total:       {total:>14,}")
    print(f"  Trainable:   {trainable:>14,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen:      {frozen:>14,}  ({100*frozen/total:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    import os
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 4))
    grad_accum = max(1, args.global_batch_size // (world_size * args.per_device_batch_size))

    print(f"=== Training Config (MathVerse / Qwen3-VL-8B / NO FEEDBACK) ===")
    print(f"  Model:              {args.model_name}")
    print(f"  Data:               {args.data_dir}")
    print(f"  WORLD_SIZE:         {world_size}")
    print(f"  per_device_batch:   {args.per_device_batch_size}")
    print(f"  grad_accum:         {grad_accum}")
    print(f"  effective_batch:    {world_size * args.per_device_batch_size * grad_accum}")
    print(f"  lr:                 {args.learning_rate}")
    print(f"  epochs:             {args.num_train_epochs}")
    print(f"  freeze_vision:      {args.freeze_vision}")
    print(f"  Trainer:            DistilTrainer (NO online feedback)")
    print(f"====================================")

    print(f"\nLoading student model from {args.model_name} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.freeze_vision:
        freeze_vision_parameters(model)

    print("Loading teacher (ref) model ...")
    teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name, min_pixels=262144, max_pixels=4194304,
    )

    print(f"\nLoading dataset from {args.data_dir}/train ...")
    dataset = load_from_disk(f"{args.data_dir}/train")

    # Print teacher type distribution
    from collections import Counter
    type_counts = Counter(dataset["teacher_type"])
    print(f"  Teacher type distribution:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c} ({100*c/len(dataset):.1f}%)")

    dataset = dataset.map(build_prompts, load_from_cache_file=False,
                          desc="Building prompts")
    print(f"Train: {len(dataset)} examples")

    r0 = dataset[0]
    print(f"  Student text: {r0['prompt'][0]['content'][1]['text'][:120]}...")
    print(f"  Teacher text: {r0['teacher_prompt'][0]['content'][1]['text'][:120]}...")

    config = DistilConfig(
        seed=args.seed,
        use_vllm=False,
        shuffle_dataset=True,
        temperature=1.0,
        top_p=0.99,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=1,
        max_prompt_length=8192,
        max_completion_length=2048,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        max_grad_norm=1.0,
        report_to=args.report_to,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        num_loss_tokens_to_skip=3,
        gradient_checkpointing=True,
        eval_strategy="no",
    )

    class EpochSaveCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
        callbacks=[EpochSaveCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()

    if trainer.accelerator.is_main_process:
        print(f"\nFinal model saved to {args.output_dir}")
