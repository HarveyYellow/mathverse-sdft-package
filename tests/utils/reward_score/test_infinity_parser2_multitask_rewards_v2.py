import json
import unittest
import importlib.util
from pathlib import Path


def load_reward_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = (
        repo_root
        / "examples"
        / "inf"
        / "reward_functions"
        / "infinity_parser2_multitask_rewards_v2.py"
    )
    spec = importlib.util.spec_from_file_location("infinity_parser2_multitask_rewards_v2", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestComputeScoreRealCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_reward_module()

    def test_table(self):
        # simple identical table html should give same teds_reward
        table_html = "<table><tr><td>1</td><td>1</td></tr><tr><td>1</td><td>1</td></tr></table>"
        direct = self.mod.teds_reward(table_html, table_html)
        res = self.mod.compute_score(table_html, table_html, data_source="table2html")
        self.assertEqual(res["score"], direct)
        self.assertEqual(res["score"], 1.0)

        # second case: different content
        table_md = "| 1 | 1 |\n| --- | --- |\n| 1 | 1 |"
        direct = self.mod.teds_reward(table_md, table_html)
        res = self.mod.compute_score(table_md, table_html, data_source="table2md")
        self.assertEqual(res["score"], direct)
        self.assertEqual(res["score"], 1.0)

        # TEDS returns 0.0 if prediction is empty while ground truth has a table
        table_html = "<table><tr><td>1</td></tr></table>"
        res = self.mod.compute_score("", table_html, data_source="table2html")
        self.assertEqual(res["score"], 0.0)

    def test_formula(self):
        # formula2latex -> cdm_reward
        sol = "$x=1$"
        gt = "$x=1$"
        res = self.mod.compute_score(sol, gt, data_source="formula2latex")
        self.assertEqual(res["score"], self.mod.cdm_reward(sol, gt))
        self.assertEqual(res["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
