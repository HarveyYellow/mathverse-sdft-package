# MathVerse 自适应反馈自蒸馏框架

基于 Qwen3-VL-8B-Instruct 的视觉数学推理自蒸馏框架，在 MathVerse 数据集上运行。系统为每道题自适应地构造 teacher prompt（正确 CoT / 错误反思 / 仅答案），并支持训练过程中的在线反馈注入。

## 目录

- [概述](#概述)
- [流程总览](#流程总览)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据生成流水线（3步）](#数据生成流水线3步)
- [训练](#训练)
- [评估](#评估)
- [核心超参数](#核心超参数)
- [文件说明](#文件说明)

---

## 概述

### 核心思路

传统知识蒸馏对所有题目使用相同的 teacher prompt。本框架根据基座模型在每道题上的**实际表现**，自适应地选择不同的 teacher prompt 策略：

| 基座模型表现 | Teacher 类型 | Teacher Prompt 包含的内容 |
|-------------|-------------|------------------------|
| 8次采样中至少1次正确 | `correct_cot` | 正确答案 + 最短的正确推理链作为参考 |
| 8次采样全部错误 | `reflection_with_answer` | 正确答案 + 基于错误解答生成的错误反思 |
| 8次全错（无有效反思） | `answer_only` | 仅正确答案 |

### 在线反馈机制

训练过程中，`FeedbackDistilTrainer` 会进一步动态调整：

1. Student 模型对题目生成一个回答
2. 系统从回答中提取 `\boxed{}` 中的答案，与标准答案对比
3. **如果 student 回答错误**，teacher prompt 会被追加以下内容：
   ```
   [Student's Previous Attempt]: The student answered "{错误答案}" but this is incorrect.
   Considering the student's mistake, construct a solution that clearly avoids this error.
   ```
4. Teacher 模型基于这个**增强后的 prompt** 生成解答
5. Student 通过 KL 散度损失向 teacher 的（更有针对性的）输出对齐

这形成了一个**闭环**：teacher 的指导在每个训练步骤中都根据 student 的具体错误动态调整。

---

## 流程总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据生成流水线                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 用基座模型对每道训练题采样8次                                │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  输入:  MathVerse 训练集（2283 道题 + 图像）              │      │
│  │  模型:  Qwen3-VL-8B-Instruct（原始未修改）                │      │
│  │  方式:  8次采样（temperature=1.0, top_p=0.9）             │      │
│  │  输出:  每道题的评估结果（正确/错误标记）                   │      │
│  └──────────────────────┬───────────────────────────────────┘      │
│                         │                                           │
│                         ▼                                           │
│       ┌─────────────────┴─────────────────┐                        │
│       │  至少1/8正确     │    0/8全部错误   │                       │
│       │  （1202道题）    │   （1043道题）    │                       │
│       └────────┬──────────┴───────┬────────┘                       │
│                │                  │                                  │
│                │                  ▼                                  │
│                │    Step 2: 为全错题生成错误反思                      │
│                │    ┌────────────────────────────────────────┐      │
│                │    │  Prompt: "审查学生的错误解答，             │      │
│                │    │    用3行指出错误"                          │      │
│                │    │  每题生成8条反思，选最短且有效的（≤150词） │      │
│                │    └──────────────────┬─────────────────────┘      │
│                │                       │                             │
│                ▼                       ▼                             │
│  Step 3: 组装训练数据集                                              │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  correct_cot（1202条）:         答案 + 最短正确CoT          │     │
│  │  reflection_with_answer（1043条）: 答案 + 错误反思          │     │
│  │  输出: HuggingFace Dataset                                 │     │
│  │        (images, question, answer, teacher_type, teacher_extra) │  │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                           训练                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Student 模型 ◄──── KL Loss ────► Teacher 模型             │     │
│  │   （可训练）           ▲              （冻结，EMA同步）       │     │
│  │       │              │                     │               │     │
│  │       ▼              │                     ▼               │     │
│  │  Student Prompt  在线反馈注入          Teacher Prompt       │     │
│  │  （纯题目文本）  （若student答错，    （答案 + 反思          │     │
│  │                  注入错误信息）        + 可选的在线反馈）    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                           评估                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Greedy:   temperature=0, n=1   → 贪心准确率               │     │
│  │  Sampling: temperature=1, n=8   → Top1-Avg / Top8          │     │
│  │  评分器:   mathruler.grader（与 GRPO 训练 reward 一致）     │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
mathverse_sdft_package/
│
├── README.md                        # 英文文档
├── README_CN.md                     # 本文件（中文文档）
│
├── core/                            # 核心训练框架
│   ├── distil_trainer.py            # DistilTrainer — 基础训练器（扩展自 TRL）
│   └── distil_config.py             # DistilConfig — 训练超参数定义
│
├── data/                            # 所有数据集
│   ├── raw/                         # 未处理的原始数据
│   │   └── sdft_mathverse/
│   │       ├── train/               # 2283条训练样本（图像 + 题目 + 答案）
│   │       └── test/                # 257条测试样本
│   │
│   ├── intermediate/                # 流水线中间产物
│   │   ├── mathverse_eval_results_vllm/        # Step 1 输出：每题8次采样评估结果
│   │   ├── mathverse_reflection_n8_truncated/  # Step 2 输出：错误反思（截断版）
│   │   └── mathverse_reflection_n8_full/       # Step 2 输出：错误反思（完整版）
│   │
│   └── processed/                   # 最终训练数据集（Step 3 输出）
│       ├── mathverse_sdft_ref_ans_trunc/    # 反思+答案 + 正确CoT（截断版）
│       ├── mathverse_sdft_ref_only_trunc/   # 仅反思，无答案（截断版）
│       ├── mathverse_sdft_ref_ans_full/     # 反思+答案 + 正确CoT（完整版）
│       ├── mathverse_sdft_ref_only_full/    # 仅反思，无答案（完整版）
│       └── mathverse_sdft_ans_only_ablation/  # 消融用：所有反思 → answer_only
│
├── data_pipeline/                   # 3步数据生成流水线
│   ├── step1_eval_mathverse.py      # Step 1: 评估基座模型（每题8次采样）
│   ├── step2_gen_reflection_mathverse.py  # Step 2: 为全错题生成错误反思
│   └── step3_build_mathverse_train.py     # Step 3: 组装最终训练数据集
│
├── training/                        # 训练入口脚本
│   ├── main_vlm_mathverse_qwen3.py              # 有在线反馈（FeedbackDistilTrainer）
│   └── main_vlm_mathverse_qwen3_no_feedback.py  # 无在线反馈（DistilTrainer，消融用）
│
├── evaluation/                      # 评估脚本
│   ├── eval_mathverse_vllm.py       # 主评估: greedy + 8次采样（vLLM 加速）
│   ├── rescore_all.py               # 用 mathruler.grader 重新评分已有结果
│   └── score_ablation.py            # 计算消融实验的 Top1-Avg 指标
│
├── eval_results/                    # 评估输出 JSON 文件 + 汇总
│   ├── mathverse/                   # SD + GRPO 各步评估结果（83个文件）
│   │   ├── base_qwen3vl8b_*.json   #   基座模型（greedy + sampling）
│   │   ├── sd_ref_ans_trunc_*.json  #   SD ref_ans_trunc ep1/ep2
│   │   ├── sd_ref_only_trunc_*.json #   SD ref_only_trunc ep1/ep2
│   │   ├── sd_ref_ans_full_*.json   #   SD ref_ans_full ep1/ep2
│   │   ├── sd_ref_only_full_*.json  #   SD ref_only_full ep1/ep2
│   │   ├── sd_ans_only_ablation_*.json  # 消融实验（无反馈）ep1/ep2
│   │   └── grpo_*.json              #   GRPO 逐步（step 1-31）+ 逐epoch
│   ├── mathverse_4plus4/            # GRPO 4+4 epoch 评估结果（16个文件）
│   ├── eval_results_with_avg.txt    # 最终汇总（Top1-Avg + Top8 + Greedy）
│   ├── eval_results_all_summary.txt # 早期汇总
│   └── eval_results_rescored.txt    # 重新评分结果
│
├── config/                          # 配置文件
│   ├── accelerate_config_gqa.yaml   # 分布式训练配置（DeepSpeed ZeRO-2，4卡）
│   └── requirements.txt             # Python 依赖
│
└── scripts/                         # 启动脚本
    ├── run_mathverse_data_pipeline.sh    # 一键运行完整数据流水线（step1→2→3）
    ├── run_all_mathverse_training.sh     # 完整训练流水线（多个SD变体）
    ├── run_mathverse_no_feedback.sh      # 无在线反馈训练（消融用）
    ├── run_ablation_ans_only.sh          # 消融实验训练（反馈 vs 无反馈）
    ├── run_eval_ablation.sh              # 评估消融实验 checkpoint
    └── run_eval_epoch_only.sh            # 批量评估所有 epoch 级 checkpoint
```

---

## 环境配置

### 硬件要求

- 4 张 NVIDIA H100（或 A100）GPU，每张 80GB 显存
- CPU 内存约 60GB（DeepSpeed 优化器卸载到 CPU）

### 软件依赖

```bash
pip install -r config/requirements.txt
```

核心依赖：
| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.9.0 | 深度学习框架 |
| `transformers` | 4.57.1 | 模型加载、分词器 |
| `trl` | 0.24.0 | 基础训练器类（GRPO/蒸馏） |
| `vllm` | 0.12.0 | 高速推理引擎（数据生成 + 评估） |
| `accelerate` | 1.11.0 | 分布式训练 |
| `deepspeed` | 0.18.4 | ZeRO-2 显存优化 |
| `mathruler` | (pip) | 答案提取与评分（与 GRPO reward 一致） |
| `qwen_vl_utils` | (pip) | Qwen-VL 图像处理 |
| `flashinfer-python` | 0.5.3 | vLLM 的 Flash Attention 后端 |

### 模型

将 Qwen3-VL-8B-Instruct 下载到本地：
```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct
```

---

## 数据生成流水线（3步）

### 前置条件

MathVerse 数据集需以 HuggingFace Dataset 格式保存在 `data/sdft_mathverse/train/`，包含以下列：
- `images`：PIL 图像列表
- `question`：题目文本（可能包含选项）
- `answer`：标准答案字符串

### Step 1：评估基座模型

对每道训练题用未修改的基座模型采样8次，根据正确率分类题目难度。

```bash
# 4个分片并行运行（每片占1张GPU）：
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step1_eval_mathverse.py \
        --shard $SHARD --num_shards 4 &
done
wait
```

**配置参数：**
- 采样：`temperature=1.0, top_p=0.9, n=8, max_tokens=2048`
- 答案匹配：`extract_boxed()` → `match_mathverse()`（支持选择题选项映射 + 数值容差）
- 输出：`mathverse_eval_results_vllm/eval_shard{0,1,2,3}.jsonl`

**输出格式**（每道题一条）：
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

### Step 2：生成错误反思

对全错题目（0/8 正确），每题生成8条结构化的错误反思。

```bash
# 截断版（默认，学生回答截断到1500字符）：
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step2_gen_reflection_mathverse.py \
        --shard $SHARD --num_shards 4 --truncate &
done
wait
```

**反思生成的 Prompt 模板：**
```
You are a math teacher reviewing a student's wrong solution.
Be brief and precise — your entire response must be under 150 words.

## Problem
{题目}

## Student's Solution (WRONG)
{学生的错误解答}

## Correct Answer
{标准答案}

## Task
用3行指出具体错误：
1. **Error Type**（错误类型，从以下选一：图像误读 / 定理误用 /
   计算错误 / 推理循环 / 变量赋值错误 / 图形关系错误 / 选项映射错误）
2. **Error Detail**：一句话精确指出推理在哪里出错
3. **Correction Hint**：一句话告诉学生如何修正
```

**配置参数：**
- 采样：`temperature=0.7, top_p=0.95, frequency_penalty=0.3, n=8, max_tokens=512`
- 输出：`mathverse_reflection_n8_truncated/reflection_shard{0,1,2,3}.jsonl`

### Step 3：组装训练数据集

将评估结果和反思合并为最终训练数据集。

```bash
python data_pipeline/step3_build_mathverse_train.py \
    --reflection_dir mathverse_reflection_n8_truncated \
    --suffix trunc
```

**分配逻辑：**

| 条件 | teacher_type | teacher_extra |
|------|-------------|---------------|
| 至少 1/8 正确 | `correct_cot` | Step 1 中最短的正确推理链 |
| 0/8 全错，有有效反思 | `reflection_with_answer` | 最短的反思（不超过150词） |
| 0/8 全错，无有效反思 | `answer_only` | 空字符串 |

**输出：** HuggingFace Dataset，保存在 `data/mathverse_sdft_ref_ans_trunc/train/`，包含列：
- `images`、`question`、`answer`、`teacher_type`、`teacher_extra`

**典型分布：** ~1202 correct_cot + ~1043 reflection_with_answer = ~2245 条

---

## 训练

### Teacher Prompt 构造方式

训练时，根据每条样本的 `teacher_type` 构造不同的 teacher prompt：

```python
# correct_cot：提供答案 + 参考推理链
"[Correct Answer]: {answer}"
"[Reference Reasoning]: {teacher_extra}"
"Given the correct answer above, construct a clear step-by-step solution..."

# reflection_with_answer：提供答案 + 错误分析
"[Correct Answer]: {answer}"
"[Error Analysis from a previous attempt]: {teacher_extra}"
"Given the correct answer and error analysis above, construct a clear step-by-step solution..."

# answer_only：仅提供答案
"[Correct Answer]: {answer}"
"Given the correct answer above, construct a clear step-by-step solution..."
```

Student prompt 始终是**纯题目文本**，不包含任何提示信息。

### 有在线反馈的训练（默认模式）

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

使用 `FeedbackDistilTrainer`：在每个训练步骤中，如果 student 的生成答案错误，teacher prompt 会被动态追加 student 的错误答案信息，使 teacher 生成更有针对性的解答。

### 无在线反馈的训练（消融实验）

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

使用基础 `DistilTrainer`：不注入在线反馈。消融实验中，数据也应去除反思（全部转为 `answer_only`）。

### 训练架构

```
Student 模型（可训练）
    ├── 输入纯题目文本，生成回答
    ├── Loss = KL(student_logits || teacher_logits)，计算在 completion tokens 上
    └── 跳过 completion 前3个 token 的 loss（num_loss_tokens_to_skip=3）

Teacher/Reference 模型（冻结，EMA 同步）
    ├── 输入增强后的 teacher prompt（含答案/反思/反馈），生成回答
    ├── EMA 同步：ref_model ← α * student + (1-α) * ref_model，每步执行
    └── α = ref_model_mixup_alpha = 0.01
```

---

## 评估

### 运行评估

```bash
# Greedy 模式（temperature=0, 单次生成）：
python evaluation/eval_mathverse_vllm.py \
    --model_path outputs/mathverse_ref_ans_trunc/checkpoint-35 \
    --processor_path /path/to/Qwen3-VL-8B-Instruct \
    --label sd_ref_ans_trunc_ep1 \
    --output_dir eval_results \
    --mode greedy

# Sampling 模式（temperature=1, 8次采样）：
python evaluation/eval_mathverse_vllm.py \
    --model_path outputs/mathverse_ref_ans_trunc/checkpoint-35 \
    --processor_path /path/to/Qwen3-VL-8B-Instruct \
    --label sd_ref_ans_trunc_ep1 \
    --output_dir eval_results \
    --mode sampling
```

### 评估指标

| 指标 | 说明 |
|------|------|
| **Greedy** | `temperature=0` 单次生成的准确率 |
| **Top1-Avg** | 每道题 `mean(正确次数 / 8)`，即8次采样的平均通过率，比单次 Top1 更稳定 |
| **Top8** | 8次采样中至少1次正确——模型能力的上界 |

所有答案匹配使用 `mathruler.grader.extract_boxed_content` + `grade_answer`，与 GRPO 训练的 reward 函数一致。

### 重新评分

```bash
# 用 mathruler.grader 重新评分所有结果：
python evaluation/rescore_all.py

# 计算消融实验的 Top1-Avg 指标：
python evaluation/score_ablation.py
```

---

## 核心超参数

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 1e-6 | 学习率 |
| `lr_scheduler_type` | cosine | 余弦退火 |
| `warmup_ratio` | 0.0 | 无预热 |
| `num_train_epochs` | 2 | 训练轮数 |
| `per_device_train_batch_size` | 1 | 单卡 batch size |
| `global_batch_size` | 64 | 有效 batch（通过梯度累积实现） |
| `max_prompt_length` | 8192 | 最大输入 token 数 |
| `max_completion_length` | 2048 | 最大生成 token 数 |
| `temperature` | 1.0 | Student 生成温度 |
| `top_p` | 0.99 | Student 生成 top-p |
| `num_generations` | 1 | 每个 prompt 每步的生成次数 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `ref_model_mixup_alpha` | 0.01 | Teacher 模型 EMA 同步率 |
| `ref_model_sync_steps` | 1 | 每 N 步同步一次 |
| `num_loss_tokens_to_skip` | 3 | 跳过 completion 前 N 个 token 的 loss |
| `gradient_checkpointing` | True | 梯度检查点（节省显存） |
| `bf16` | True | 混合精度 |

### 分布式训练（DeepSpeed ZeRO-2）

| 参数 | 值 |
|------|-----|
| `zero_stage` | 2 |
| `offload_optimizer_device` | cpu（优化器状态卸载到 CPU） |
| `offload_param_device` | none（模型参数不卸载） |
| `num_processes` | 4（4卡数据并行） |
| `mixed_precision` | bf16 |

### 数据生成流水线

| 步骤 | 关键参数 |
|------|---------|
| Step 1（评估） | `temperature=1.0, top_p=0.9, n=8, max_tokens=2048` |
| Step 2（生成反思） | `temperature=0.7, top_p=0.95, frequency_penalty=0.3, n=8, max_tokens=512` |
| Step 3（组装） | 选最短的正确 CoT；选最短的反思（不超过150词） |

---

## 文件说明

### `core/distil_trainer.py`

扩展自 TRL 的 `BaseTrainer`，实现自蒸馏训练循环：
- Student 用纯题目 prompt 生成 completion
- Teacher 用增强 prompt（含答案/反思）生成 completion
- 计算 student 和 teacher logits 之间的 KL 散度损失
- Teacher 权重通过 EMA 向 student 同步
- 支持视觉语言模型（VLM）的多模态输入

### `core/distil_config.py`

扩展自 `TrainingArguments`，添加蒸馏专用字段：
- `num_generations`、`temperature`、`top_p`：生成控制
- `max_prompt_length`、`max_completion_length`：序列长度限制
- `sync_ref_model`、`ref_model_sync_steps`、`ref_model_mixup_alpha`：EMA 同步控制
- `num_loss_tokens_to_skip`：跳过 completion 开头 token 的损失计算

### `training/main_vlm_mathverse_qwen3.py`

训练入口，使用 `FeedbackDistilTrainer`（继承自 `DistilTrainer`）：
- `_generate_and_score_completions()`：student 生成后，检查答案是否正确；若错误，在 teacher prompt 中注入 `[Student's Previous Attempt]`
- `_extract_boxed()`：正则提取 `\boxed{}` 中的内容
- `_answers_match()`：字符串匹配 + 浮点数容差匹配
- `_generate_single_turn()`：OOM 安全的生成，捕获显存溢出后回退到 `max_new_tokens=1`
- 记录 `feedback/wrong_ratio` 指标（每个 batch 中触发反馈注入的比例）

### `training/main_vlm_mathverse_qwen3_no_feedback.py`

与上述相同，但直接使用基础 `DistilTrainer`。不包含 `FeedbackDistilTrainer` 类，不进行在线反馈注入。用于消融实验，隔离反馈机制的效果。

### `evaluation/eval_mathverse_vllm.py`

使用 vLLM 进行高速批量推理评估：
- 支持 `greedy`（temperature=0, n=1）和 `sampling`（temperature=1, n=8）两种模式
- 使用 `mathruler.grader` 进行答案提取和匹配（与 GRPO 训练 reward 一致）
- 保存完整的模型输出，便于事后分析
- 跳过逻辑：若输出文件已存在，自动跳过
- vLLM 配置：`tensor_parallel_size=4, gpu_memory_utilization=0.85, max_model_len=8192`

### `evaluation/rescore_all.py`

用 `mathruler.grader` 重新评分所有已有的评估结果 JSON 文件，统一评分标准。

### `evaluation/score_ablation.py`

专门为消融实验计算 Top1-Avg 指标：对每道题的 8 次采样结果计算 `正确次数/8` 的平均值，比单次抽样的 Top1 更稳定可靠。
