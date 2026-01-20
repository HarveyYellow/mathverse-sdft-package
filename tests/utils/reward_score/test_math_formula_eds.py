import importlib.util
import unittest
from pathlib import Path


def load_reward_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = (
        repo_root
        / "examples"
        / "inf"
        / "reward_functions"
        / "math_formula_eds.py"
    )
    spec = importlib.util.spec_from_file_location("math_formula_eds", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestNormalizeLatex(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_reward_module()

    def test_normalize_examples(self):
        test_cases = [
            (
                r"x_{i}^{2} + \left( \frac{1}{2} \right) \to \infty",
                r"x^2_i + \frac{1}{2} \rightarrow \infty",
            ),
            (r"\left(  a  \quad + \quad b  \right) = \Big[ c \Big]", r"(a+b)=[c]"),
            (r"a \le b \to c \ast d \ge e", r"a\leqb\rightarrowc*d\geqe"),
            (r"x^2_i + y^{abc}_{def} + z^k_{n}", r"x_i^2+y_{def}^{abc}+z_n^k"),
            (r"e^{x} + a_{1} + b_{12} + c^{ \pi }", r"e^x+a_1+b_{12}+c^{\pi}"),
            (
                r"\sum ^{ \infty }_{ n = 1 } \frac { 1 } { n^2 } \to \frac{\pi^2}{6}",
                r"\sum_{n=1}^\infty\frac{1}{n^2}\rightarrow\frac{\pi^2}{6}",
            ),
        ]

        for idx, (pd_inp, gt_inp) in enumerate(test_cases):
            with self.subTest(case=idx):
                pd_out = self.mod.normalize_latex(pd_inp)
                gt_out = self.mod.normalize_latex(gt_inp)

                self.assertEqual(
                    pd_out,
                    gt_out,
                    msg=f"Case {idx} failed: pd_out={pd_out!r} gt_out={gt_out!r}",
                )


if __name__ == "__main__":
    unittest.main()
