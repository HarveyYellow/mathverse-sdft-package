import os
import sys
from mailbox import NotEmptyError
import re
from typing import List, Dict, Tuple

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(cur_path))

from examples.inf.reward_functions.doc2json import Doc2JsonReward
from examples.inf.reward_functions.eds import accuracy_reward as eds_reward
from examples.inf.reward_functions.teds import TEDS
from examples.inf.reward_functions.teds import teds_reward
from examples.inf.reward_functions.cdm import cdm_reward
from examples.inf.reward_functions.bleu import bleu_reward
from examples.inf.reward_functions.rmsf1 import rmsf1_reward
from examples.inf.reward_functions.scrm import scrm_reward
from examples.inf.reward_functions.tanimoto import tanimoto_reward
from examples.inf.reward_functions.anls import anls_reward


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
        accuracy_score = teds_reward(solution_str, ground_truth)
    elif data_source == "formula2latex":
        accuracy_score = cdm_reward(solution_str, ground_truth)
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

    result = {
        "score": accuracy_score,
        "format": 0,
        "accuracy": accuracy_score,
    }
    return result
