"""V3 Contrastive Visual Feedback: Prompt templates.

Design philosophy:
  V1 (specific): "you made error X, fix it by doing Y" → shortcut
  V2 (general): "be careful about X" → too vague, model confirms its own bias
  V3 (contrastive visual): pinpoint the exact visual element where correct
      and incorrect interpretations diverge → model must re-examine the image

Two-stage pipeline:
  Stage A: Generate contrastive visual feedback (pinpoint visual contradiction)
  Stage B: Use feedback to re-solve (with contrastive framing)
"""

# ======================== Stage A: Feedback Generation ========================

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


# ======================== Stage B: Answering with Contrastive Feedback ========================

ANSWER_SYSTEM_PROMPT = ("Solve this problem. Put your final answer in \\boxed{}. "
                        "If the question provides multiple-choice options, "
                        "only put the option letter in the box.")

ANSWER_FEEDBACK_TEMPLATE = """[Before solving, check this about the image]:

A key visual element can be read two ways:
{feedback}

Verify in the image which reading is correct, then solve."""
