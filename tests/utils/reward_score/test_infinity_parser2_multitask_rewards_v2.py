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
        # table2html
        sol = "<table><tr><td><b>Sample</b></td><td><b><i>T</i> (°C)</b></td><td><b><i>D</i> (nm)</b></td><td><b><i>T</i><sub>C</sub> (°C)</b></td><td><b><i>M</i><sub>S</sub> (emu/g)</b></td><td><b><i>M</i><sub>r</sub> (emu/g)</b></td><td><b><i>H</i><sub>C</sub> (Oe)</b></td><td><b><i>t</i>/<i>D</i> (%)</b></td></tr><tr><td>a</td><td>240</td><td>23</td><td>335</td><td>40.47</td><td>6.81</td><td>156</td><td>4.26</td></tr><tr><td>b</td><td>255</td><td>45</td><td>346</td><td>44.33</td><td>9.67</td><td>119</td><td>3.09</td></tr><tr><td>c</td><td>270</td><td>80</td><td>351</td><td>46.47</td><td>12.56</td><td>76</td><td>2.43</td></tr><tr><td>d</td><td>285</td><td>114</td><td>354</td><td>47.80</td><td>17.59</td><td>17</td><td>2.03</td></tr></table>"
        gt = "<table><tr><td><b>Sample</b></td><td><b><i>T</i>(°C)</b></td><td><b><i>D</i>(nm)</b></td><td><b><i>T</i><sub>C</sub>(°C)</b></td><td><b><i>M</i><sub>S</sub>(emu/g)</b></td><td><b><i>M</i><sub>r</sub>(emu/g)</b></td><td><b><i>H</i><sub>C</sub>(Oe)</b></td><td><b><i>t</i>/<i>D</i>(%)</b></td></tr><tr><td>a</td><td>240</td><td>23</td><td>335</td><td>40.47</td><td>6.81</td><td>156</td><td>4.26</td></tr><tr><td>b</td><td>255</td><td>45</td><td>346</td><td>44.33</td><td>9.67</td><td>119</td><td>3.09</td></tr><tr><td>c</td><td>270</td><td>80</td><td>351</td><td>46.47</td><td>12.56</td><td>76</td><td>2.43</td></tr><tr><td>d</td><td>285</td><td>114</td><td>354</td><td>47.80</td><td>17.59</td><td>17</td><td>2.03</td></tr></table>"
        direct = self.mod.teds_reward(sol, gt)
        res = self.mod.compute_score(sol, gt, data_source="table2html")
        self.assertEqual(res["score"], direct)
        self.assertGreater(res["score"], 0.95)

        # table2md
        sol = "```markdown\n|  |  | the |  |\n| --- | --- | --- | --- |\n|  | Note | form | stabilise |\n| rather than having an internal In why does so |  | -6692.68 | 5554.32 |\n| users with physical disabilities a |  |  |  |\n| parasitoids and predators |  | -5228.19 |  |\n| cannot be readily The depletion of these |  |  | 4497.94 |\n| language spoken on Aruba Bonaire and Curaao that |  | 4440.62 | 4265.55 |\n| please the an appeal is the |  | 8572.83 | 2231.51 |\n| how ASD is viewed |  |  | -4296.5 |\n| yogurt and Kabuli palaw is |  | -4732.46 | -6676.07 |\n| important type of aquaculture in some | -7764.99 | -3539.5 | 2441.89 |\n| of a trial may |  | 8049.27 | 2692.82 |\n| was the nemesis of many artists during the |  | 4279.49 | -2119.31 |\n| their numbers are Armenian was the majority |  |  | -9316.58 |\n| lead to hyperkalemia strongly influencing the |  | -9936.75 |  |\n| but this list was derived from analysis of It | -2748.94 |  | 6250.6 |\n| followed suit and entered winter quarters |  |  | -5996.11 |\n```"
        gt = "```markdown\n|  |  | the |  |\n| --- | --- | --- | --- |\n|  | Note | form | stabilise |\n| rather than having an internal In why does so |  | -6692.68 | 5554.32 |\n| users with physical disabilities a |  |  |  |\n| parasitoids and predators |  | -5228.19 |  |\n| cannot be readily The depletion of these |  |  | 4497.94 |\n| language spoken on Aruba Bonaire and Curaao that |  | 4440.62 | 4265.55 |\n| please the an appeal is the |  | 8572.83 | 2231.51 |\n| how ASD is viewed |  |  | -4296.5 |\n| yogurt and Kabuli palaw is |  | -4732.46 | -6676.07 |\n| important type of aquaculture in some | -7764.99 | -3539.5 | 2441.89 |\n| of a trial may |  | 8049.27 | 2692.82 |\n| was the nemesis of many artists during the |  | 4279.49 | -2119.31 |\n| their numbers are Armenian was the majority |  |  | -9316.58 |\n| lead to hyperkalemia strongly influencing the |  | -9936.75 |  |\n| but this list was derived from analysis of It | -2748.94 |  | 6250.6 |\n| followed suit and entered winter quarters |  |  | -5996.11 |\n```"
        direct = self.mod.teds_reward(sol, gt)
        res = self.mod.compute_score(sol, gt, data_source="table2md")
        self.assertEqual(res["score"], direct)
        self.assertGreater(res["score"], 0.95)

        # TEDS returns 0.0 if prediction is empty while ground truth has a table
        res = self.mod.compute_score("", gt, data_source="table2md")
        self.assertEqual(res["score"], 0.0)

    def test_formula(self):
        # formula2latex -> cdm_reward
        sol = "$$\\gamma_{0}^{\\prime}\\approx-h_{1}\\gamma_{0}^{3}+O(\\gamma_{0}^{5}),$$"
        gt = "$$\\gamma_{0}^{\\prime}\\approx-h_{1}\\gamma_{0}^{3}+{\\cal{O}}(\\gamma_{0}^{5}),$$"
        res = self.mod.compute_score(sol, gt, data_source="formula2latex")
        self.assertEqual(res["score"], self.mod.cdm_reward(sol, gt))
        self.assertEqual(res["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
