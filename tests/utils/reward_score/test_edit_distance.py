import unittest

# from unittest.mock import patch, MagicMock
import os
import sys
import json
import random

random.seed(42)  # You can use any integer value as the seed
import copy
from collections import defaultdict

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{cur_path}/..")

from examples.inf.reward_functions.edit_distance import compute_score


class TestRewardFunction(unittest.TestCase):

    def test_compute_score(self):
        """Test reward function."""

        response_strs = [
            "以下是解析结果：```markdown\n| 123 |\n```",
            "以下是解析结果：```html\n<tr> <td> 123 </td> </tr>\n```",
            "以下是解析结果：```latex\n$$a^2 + b^2 = c^2$$\n```",
        ]

        ground_truths = [
            "```markdown\n| 123 |\n```",
            "```html\n<tr><td>123</td></tr>\n```",
            "$$a^2 + b^2 = c^2$$",
        ]

        for response_str, ground_truth in zip(response_strs, ground_truths):
            score = compute_score(
                data_source="document_parsing",
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=None,
            )

            reward_extra_info = defaultdict(list)
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            self.assertEqual(reward, 1.0)


if __name__ == "__main__":
    unittest.main()
