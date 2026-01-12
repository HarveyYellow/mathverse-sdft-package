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
from cdm_metric import CDM
import uuid
import logging


def extract_valid_content(text):

    patterns = [
        r"```markdown\n(.*?)\n```",
        r"```html\n(.*?)\n```",
        r"```latex\n(.*?)\n```",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def cdm_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        cdm_obj = CDM(output_root="./tmp")
        reward = cdm_obj.evaluate(ground_truth, solution_str, str(uuid.uuid4()))[
            "F1_score"
        ]
        return reward
    except Exception as e:
        return 0.0
