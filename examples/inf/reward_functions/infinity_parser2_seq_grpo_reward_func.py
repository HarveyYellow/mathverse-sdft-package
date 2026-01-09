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
import sys
import os
import json
from typing import List, Tuple, Dict

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from doc2json import DocumentParserReward


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = None,
    extra_info: str | dict = None,
    format_score: float = 0.3,
) -> dict:
    reward_calculator = DocumentParserReward(text_weight=0.7, layout_weight=0.3)
    reward, _, _ = reward_calculator(solution_str, ground_truth)
    total_reward, text_reward, layout_reward = reward
    if reward == 0:
        format_reward = 0
    else:
        format_reward = 1
    result = {
        "score": total_reward,
        "format": format_reward,
        "text_reward": text_reward,
        "layout_reward": layout_reward,
    }
    return result
