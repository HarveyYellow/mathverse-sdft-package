# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Dict, Tuple
import Levenshtein
import traceback


def anls(
    pred,
    references,
    thresh_hold=0.5,
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []

    for answer in references:
        # preprocess both the answers - gt and prediction
        det_answer = " ".join(pred.strip().lower().split())
        gt_answer = " ".join(answer.strip().lower().split())

        dist = Levenshtein.distance(det_answer, gt_answer)
        length = max(len(pred.upper()), len(answer.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return question_result


def anls_reward(solution_str: str, ground_truth: List[str]) -> float:
    try:
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        reward = anls(solution_str, ground_truth)
        return reward
    except Exception as e:
        traceback.print_exc()
        return 0.0
