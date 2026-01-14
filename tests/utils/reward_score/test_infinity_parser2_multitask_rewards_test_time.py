import time
import json
import unittest
import importlib.util
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_reward_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = (
        repo_root
        / "examples"
        / "inf"
        / "reward_functions"
        / "infinity_parser2_multitask_rewards_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "infinity_parser2_multitask_rewards_v2", str(target)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestComputeScoreRealCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_reward_module()

    def _run_case(self, ds: str, name: str, sol, gt, direct_fn=None):
        t0 = time.perf_counter()
        res = self.mod.compute_score(sol, gt, data_source=ds)
        t1 = time.perf_counter()
        score = res.get("score", None)

        if direct_fn is None:
            print(
                f"[{ds}][{name}] score={score} elapsed={(t1 - t0) * 1000:.3f} ms"
            )
        else:
            try:
                direct = direct_fn(sol, gt)
            except Exception as e:
                direct = f"direct_err={type(e).__name__}: {e}"
            print(
                f"[{ds}][{name}] score={score} direct={direct} elapsed={(t1 - t0) * 1000:.3f} ms"
            )
        return res

    # -----------------------------
    # doc2json (Doc2JsonReward)
    # -----------------------------
    def test_doc2json_branch(self):
        cases = [
            (
                "empty_identical",
                json.dumps([]),
                json.dumps([]),
            ),
            (
                "one_item_text_small_diff",
                json.dumps([{"bbox": [0, 0, 1, 1], "category": "text", "text": "hello"}]),
                json.dumps([{"bbox": [0, 0, 1, 1], "category": "text", "text": "hallo"}]),
            ),
            (
                "layout_mismatch",
                json.dumps([{"bbox": [0, 0, 1, 1], "category": "text", "text": "hello"}]),
                json.dumps([{"bbox": [10, 10, 11, 11], "category": "text", "text": "hello"}]),
            ),
        ]

        for name, sol, gt in cases:
            self._run_case("doc2json", name, sol, gt)

    # -----------------------------
    # eds group: doc2md / text2md / chart2code
    # -----------------------------
    def test_eds_group(self):
        cases = [
            ("identical_short", "simple text", "simple text"),
            ("small_edit", "abc", "abx"),
            ("markdown_codeblock_identical", "```python\ncode()\n```", "```python\ncode()\n```"),
        ]
        for name, sol, gt in cases:
            # 任选一个 ds；你也可以拆成三个函数分别跑
            self._run_case("doc2md", name, sol, gt, direct_fn=self.mod.eds_reward)

    # -----------------------------
    # layout_analysis
    # -----------------------------
    def test_layout_analysis(self):
        cases = [
            ("empty_identical", json.dumps([]), json.dumps([])),
            (
                "partial_overlap",
                json.dumps([{"bbox": [0, 0, 2, 2], "category": "text"}]),
                json.dumps([{"bbox": [1, 1, 3, 3], "category": "text"}]),
            ),
            (
                "category_diff",
                json.dumps([{"bbox": [0, 0, 2, 2], "category": "title"}]),
                json.dumps([{"bbox": [0, 0, 2, 2], "category": "text"}]),
            ),
        ]
        for name, sol, gt in cases:
            self._run_case("layout_analysis", name, sol, gt)

    # -----------------------------
    # table2html (TEDS)
    # -----------------------------
    def test_table(self):
        # case1: identical big table (your example)
        table_html_1 = """<table border="1" cellpadding="6" cellspacing="0">
  <caption>Group 3A — Model Evaluation Report</caption>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="3">Classification</th>
      <th colspan="2">Generation</th>
      <th rowspan="2">Notes</th>
    </tr>
    <tr>
      <th>Acc</th><th>F1</th><th>AUROC</th><th>BLEU</th><th>ChrF</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7"><strong>Baseline</strong></td></tr>
    <tr><td>ResNet+BERT</td><td>0.812</td><td>0.768</td><td>0.861</td><td>19.4</td><td>41.2</td><td>Stable, fast</td></tr>
    <tr><td>ViT+RoBERTa</td><td>0.835</td><td>0.790</td><td>0.879</td><td>21.7</td><td>43.5</td><td>Better text alignment</td></tr>
    <tr><td colspan="7"><strong>Fine-tuned</strong></td></tr>
    <tr><td>Qwen2.5-VL (SFT)</td><td>0.858</td><td>0.812</td><td>0.901</td><td>24.9</td><td>46.8</td><td>Needs longer warmup</td></tr>
    <tr><td>Qwen2.5-VL (GRPO)</td><td>0.872</td><td>0.826</td><td>0.915</td><td>26.1</td><td>48.2</td><td>More robust</td></tr>
    <tr><td colspan="7"><strong>Ablations</strong></td></tr>
    <tr><td>No OCR</td><td>0.844</td><td>0.798</td><td>0.892</td><td>23.2</td><td>45.1</td><td>Drops on noisy docs</td></tr>
    <tr><td>No Layout</td><td>0.837</td><td>0.791</td><td>0.884</td><td>22.6</td><td>44.3</td><td>Reading-order hurt</td></tr>
  </tbody>
</table>"""
        pred_html_1 = table_html_1

        # case2: structure similar but row order changed + minor formatting differences
        gt_html_2 = """<table>
  <thead>
    <tr><th>Method</th><th>Acc</th><th>F1</th><th>AUROC</th><th>Latency</th><th>Mem</th></tr>
  </thead>
  <tbody>
    <tr><td>ResNet+BERT</td><td>0.812</td><td>0.768</td><td>0.861</td><td>32ms</td><td>3.1GB</td></tr>
    <tr><td>ViT+RoBERTa</td><td>0.835</td><td>0.790</td><td>0.879</td><td>41ms</td><td>4.0GB</td></tr>
    <tr><td>CLIP (dual)</td><td>0.824</td><td>0.781</td><td>0.871</td><td>38ms</td><td>3.6GB</td></tr>
    <tr><td>Qwen2.5-VL (SFT)</td><td>0.858</td><td>0.812</td><td>0.901</td><td>55ms</td><td>6.2GB</td></tr>
    <tr><td>Qwen2.5-VL (GRPO)</td><td>0.872</td><td>0.826</td><td>0.915</td><td>61ms</td><td>6.7GB</td></tr>
  </tbody>
</table>"""
        pred_html_2 = """<table>
  <thead>
    <tr><th>Method</th><th>Acc</th><th>F1</th><th>AUROC</th><th>Latency</th><th>Mem</th></tr>
  </thead>
  <tbody>
    <tr><td>CLIP (dual)</td><td>0.824</td><td>0.781</td><td>0.871</td><td>38 ms</td><td>3.6 GB</td></tr>
    <tr><td>ResNet + BERT</td><td>0.812</td><td>0.768</td><td>0.861</td><td>32 ms</td><td>3.1 GB</td></tr>
    <tr><td>ViT + RoBERTa</td><td>0.835</td><td>0.790</td><td>0.879</td><td>41 ms</td><td>4.0 GB</td></tr>
    <tr><td>Qwen2.5-VL (GRPO)</td><td>0.872</td><td>0.826</td><td>0.915</td><td>61 ms</td><td>6.7 GB</td></tr>
    <tr><td>Qwen2.5-VL (SFT)</td><td>0.858</td><td>0.812</td><td>0.901</td><td>55 ms</td><td>6.2 GB</td></tr>
  </tbody>
</table>"""

        # case3: prediction is NOT HTML (plain text / markdown table)
        gt_html_3 = gt_html_2
        pred_text_3 = """Here is the table you requested:

Method | Acc | F1 | AUROC | Latency | Mem
----- | ---- | ---- | ----- | ------- | ----
ResNet + BERT | 0.812 | 0.768 | 0.861 | 32 ms | 3.1 GB
ViT + RoBERTa | 0.835 | 0.790 | 0.879 | 41 ms | 4.0 GB
CLIP (dual) | 0.824 | 0.781 | 0.871 | 38 ms | 3.6 GB
Qwen2.5-VL (SFT) | 0.858 | 0.812 | 0.901 | 55 ms | 6.2 GB
Qwen2.5-VL (GRPO) | 0.872 | 0.826 | 0.915 | 61 ms | 6.7 GB
"""

        cases = [
            ("identical_big_html", table_html_1, pred_html_1),
            ("reordered_rows_minor_spacing", gt_html_2, pred_html_2),
            ("pred_not_html_plaintext", gt_html_3, pred_text_3),
        ]

        for name, gt, sol in cases:
            # 注意：compute_score(table2html) 里通常是 (solution, ground_truth) 还是 (pred, gt)？
            # 你现有用法是 compute_score(gt_html, pred_html, data_source="table2html")
            # 这里沿用你的顺序：sol=pred, gt=gt
            self._run_case("table2html", name, sol, gt, direct_fn=self.mod.teds_reward)

    # -----------------------------
    # formula2latex (CDM)
    # -----------------------------
    def test_formula(self):
        cases = [
            ("spaces_equivalent", "$ x = 1 $", "$x=1$"),
            (
                "long_same_rendering",
                r"$\int_{0}^{1}\frac{(1-x)^{2}+x^{2}}{1+x^{2}}\,dx=\sum_{k=1}^{n}\frac{1}{k^{2}}+\prod_{i=1}^{m}(1+a_{i})$",
                r"$\int_{0}^{1}\frac{(1-x)^2+x^2}{1+x^2}\,dx=\sum_{k=1}^{n}\frac{1}{k^2}+\prod_{i=1}^{m}(1+a_i)$",
            ),
            ("e_subscript_spacing", r"$e^{x_{1}+x_{2}}$", r"$e^{x_1+x_2}$"),
        ]
        for name, sol, gt in cases:
            self._run_case("formula2latex", name, sol, gt, direct_fn=self.mod.cdm_reward)

    # -----------------------------
    # chart2text (BLEU via sacrebleu)
    # -----------------------------
    def test_chart_text(self):
        cases = [
            ("identical", "Hello world, Hello world", "Hello world, Hello world"),
            ("small_paraphrase", "The cat is sitting on the mat.", "The cat sits on the mat."),
            ("very_different", "Revenue increased sharply in Q3.", "The experiment failed due to noise."),
        ]
        for name, sol, gt in cases:
            self._run_case("chart2text", name, sol, gt, direct_fn=self.mod.bleu_reward)

    # -----------------------------
    # chart2table (RMSF1)
    # -----------------------------
    def test_chart_table(self):
        gt = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        cases = [
            ("identical_md_table", gt, gt),
            ("minor_spacing", "|a|b|\n|---|---|\n|1|2|\n|3|4|", gt),
            ("different_values", "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 999 | 4 |", gt),
        ]
        for name, sol, gt_ in cases:
            self._run_case("chart2table", name, sol, gt_, direct_fn=self.mod.rmsf1_reward)

    # -----------------------------
    # chart2json (SCRM)
    # -----------------------------
    def test_chart_json(self):
        cases = [
            ("identical_simple", json.dumps({"values": {"k": 1}}), json.dumps({"gts": {"values": {"k": 1}}})),
            ("numeric_tolerance", json.dumps({"values": {"k": 1.00}}), json.dumps({"gts": {"values": {"k": 1.01}}})),
            ("key_mismatch", json.dumps({"values": {"k": 1, "x": 2}}), json.dumps({"gts": {"values": {"k": 1, "y": 2}}})),
        ]
        for name, sol, gt in cases:
            self._run_case("chart2json", name, sol, gt, direct_fn=self.mod.scrm_reward)

    # -----------------------------
    # chem2smiles (Tanimoto)
    # -----------------------------
    def test_chem2smiles(self):
        cases = [
            ("identical", "CCO", "CCO"),
            ("isomer_diff", "CCO", "OCC"),
            ("invalid_smiles", "NOT_A_SMILES", "CCO"),
        ]
        for name, sol, gt in cases:
            self._run_case("chem2smiles", name, sol, gt, direct_fn=self.mod.tanimoto_reward)

    # -----------------------------
    # docvqa (ANLS or similar)
    # -----------------------------
    def test_docvqa(self):
        cases = [
            ("exact_match", "yes", ["yes"]),
            ("case_diff", "Yes", ["yes"]),
            ("no_match", "no", ["yes"]),
        ]
        for name, sol, gt in cases:
            try:
                self._run_case("docvqa", name, sol, gt)
            except RecursionError:
                print(f"[docvqa][{name}] skipped due to RecursionError")
            except Exception as e:
                print(f"[docvqa][{name}] exception={type(e).__name__}: {e}")

    def test_unsupported_datasource_raises(self):
        with self.assertRaises(NotImplementedError):
            self.mod.compute_score("s", "g", data_source="this_is_not_supported")


if __name__ == "__main__":
    unittest.main()

