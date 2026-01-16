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


def extract_valid_content(text):

    patterns = [
        r"```markdown\n(.*?)\n```",
        r"```html\n(.*?)\n```",
        r"```latex\n(.*?)\n```",
        r"```json\n(.*?)\n```",
        r"```python\n(.*?)\n```",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def similarity(pred: str, gt: str):
    edit_distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    normalized_distance = edit_distance / max_len if max_len > 0 else 1
    return 1.0 - normalized_distance


def accuracy_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        reward = similarity(solution_str, ground_truth)
        return reward
    except Exception as e:
        traceback.print_exc()
        return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = None,
    extra_info: str | dict = None,
    format_score: float = 0.3,
) -> dict:
    accuracy_score = accuracy_reward(solution_str, ground_truth)
    result = {
        "score": accuracy_score,
        "format": 0,
        "accuracy": accuracy_score,
    }
    return result
