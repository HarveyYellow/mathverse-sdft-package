import os
import sys
from mailbox import NotEmptyError
import re
from typing import List, Dict, Tuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, "{}/../../..".format(current_dir))

    from examples.inf.reward_functions.doc2json import Doc2JsonReward
    from examples.inf.reward_functions.eds import accuracy_reward as eds_reward
    # from examples.inf.reward_functions.teds import TEDS
    from examples.inf.reward_functions.teds import teds_reward
    from examples.inf.reward_functions.math_formula_eds import math_formula_eds_reward
    from examples.inf.reward_functions.bleu import bleu_reward
    from examples.inf.reward_functions.rmsf1 import rmsf1_reward
    from examples.inf.reward_functions.scrm import scrm_reward
    from examples.inf.reward_functions.tanimoto import tanimoto_reward
    from examples.inf.reward_functions.anls import anls_reward
except ImportError:
    # 添加当前目录到 Python 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from doc2json import Doc2JsonReward
    from eds import accuracy_reward as eds_reward
    # from teds import TEDS
    from teds import teds_reward
    # from math_formula_cdm import cdm_reward
    from bleu import bleu_reward
    from rmsf1 import rmsf1_reward
    from scrm import scrm_reward
    from tanimoto import tanimoto_reward
    from anls import anls_reward


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = None,
    extra_info: str | dict = None,
    format_score: float = 0.3,
) -> dict:
    if data_source == "doc2json":
        doc2json_reward_calculator = Doc2JsonReward(text_weight=0.7, layout_weight=0.3)
        reward, _, _ = doc2json_reward_calculator(solution_str, ground_truth)
        accuracy_score = reward[0]
    elif (
        data_source == "doc2md"
        or data_source == "text2md"
        or data_source == "chart2code"
    ):
        accuracy_score = eds_reward(solution_str, ground_truth)
    elif data_source == "layout_analysis":
        doc2json_reward_calculator = Doc2JsonReward(text_weight=0.0, layout_weight=1.0)
        reward, _, _ = doc2json_reward_calculator(solution_str, ground_truth)
        accuracy_score = reward[0]
    elif data_source == "table2html" or data_source == "table2md":
        # accuracy_score = teds_reward(solution_str, ground_truth)
        accuracy_score = eds_reward(solution_str, ground_truth)
    elif data_source == "formula2latex":
        accuracy_score = math_formula_eds_reward(solution_str, ground_truth)
    elif data_source == "chart2text":
        accuracy_score = bleu_reward(solution_str, ground_truth)
    elif data_source == "chart2table":
        accuracy_score = rmsf1_reward(solution_str, ground_truth)
    elif data_source == "chart2json":
        accuracy_score = scrm_reward(solution_str, ground_truth)
    elif data_source == "chem2smiles":
        accuracy_score = tanimoto_reward(solution_str, ground_truth)
    elif data_source == "docvqa":
        accuracy_score = anls_reward(solution_str, ground_truth)
    else:
        raise NotImplementedError

    # debug
    logger.info(f"solution_str: {solution_str}, ground_truth: {ground_truth}, data_source: {data_source}, accuracy_score: {accuracy_score}")

    result = {
        "score": accuracy_score,
        "format": 0,
        "accuracy": accuracy_score,
    }
    return result
