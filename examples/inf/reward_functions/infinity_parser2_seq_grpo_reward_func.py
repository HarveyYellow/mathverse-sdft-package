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

from infinity_parser2_seq_grpo_reward_base import extract_json_content, truncate_last_incomplete_element, DocumentParserReward


def process_single_result(pred, ground_truth, reward_calculator):
        """处理单个结果的函数"""
        res_content = extract_json_content(pred)
        new_res, trunc = truncate_last_incomplete_element(res_content)
        gt = json.loads(extract_json_content(ground_truth))
        try:
            res_parsed = json.loads(new_res)
            reward = reward_calculator(gt, res_parsed)
            return reward, trunc, None
        except Exception as e:
            return 0, trunc, (e, new_res)


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = None,
    extra_info: str | dict = None,
    format_score: float = 0.3,
) -> dict:
    reward_calculator = DocumentParserReward(text_weight=0.7, layout_weight=0.3)
    reward, _, _ = process_single_result(solution_str, ground_truth, reward_calculator)
    text_reward, layout_reward = 0, 0
    if reward == 0:
        format_reward = 0
    else:
        format_reward = 1
        text_reward = reward[1]
        layout_reward = reward[2]
    result = {
        "score": 0.7 * text_reward + 0.3 * layout_reward,
        "format": format_reward,
        "text_reward": text_reward,
        "layout_reward": layout_reward
    }
    return result
