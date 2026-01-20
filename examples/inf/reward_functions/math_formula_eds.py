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
from typing import List, Dict, Tuple, Union
import Levenshtein
import traceback


def extract_valid_content(text):

    patterns = [
        r"```latex\n(.*?)\n```",
        r"\$\$(.*?)\$\$",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def normalize_latex(latex_string):
    """
    将 LaTeX 字符串标准化，以便进行 Edit Distance 计算。
    """
    # 1. 预处理：转为 raw string 并去除前后空白
    s = latex_string.strip()

    # 2. 移除特定环境（可选，视数据集而定）
    # s = re.sub(r'\\begin\{.*?\}', '', s)
    # s = re.sub(r'\\end\{.*?\}', '', s)

    # 3. 移除由 \left 和 \right 组成的修饰（只移除命令本身，保留实际的括号/分隔符）,
    # 避免误删例如 \rightarrow 中的 'right'
    s = re.sub(r"\\left(?=[\(\[\{\.])", "", s)
    s = re.sub(r"\\right(?=[\)\]\}\.])", "", s)

    # 4. 移除由 \big, \Big 等组成的大小修饰
    s = re.sub(r"\\[bB]igg?[lr]?", "", s)

    # 5. 移除排版空格和微调间距
    # 匹配 \, \; \: \! \quad \qquad 以及普通空格
    # 注意：如果你的公式包含 \text{...}，这里需要更复杂的解析来保留 text 内部的空格
    s = re.sub(r"\\[,;!: ]|\\quad|\\qquad|\s+", "", s)

    # 6. 命令同义词替换 (Map to Canonical forms)
    # 使用正则避免对已正规化的命令造成二次替换（例如不要把 \leq 变成 \leqq）
    # Replace common commands; allow \\to replacement even if letters follow because spaces
    # are removed earlier (e.g., '\to c' -> '\toc').
    s = re.sub(r"\\to", r"\\rightarrow", s)
    s = re.sub(r"\\le(?!q)", r"\\leq", s)
    s = re.sub(r"\\ge(?!q)", r"\\geq", s)
    s = re.sub(r"\\ast", "*", s)

    # 7. 处理上下标顺序 (Normalize Sub/Superscript Order)
    # 目标：将 x^a_b 转换为 x_b^a (先下后上) 或者反之
    # 这里使用正则来捕获 pattern: ^{...}_{...} 并交换
    # 注意：这只能处理简单的单层结构，嵌套结构需要真正的 Parser，但对大多数 OCR 结果足够

    # 能够匹配 ^{...} 或 ^\w
    # Pattern explanation:
    # \^           匹配上标符号
    # (?:          非捕获组开始
    #   \{([^{}]+)\}   匹配 {content}
    #   |              或者
    #   ([a-zA-Z0-9])  匹配单字符
    # )            组结束
    # 同样的逻辑匹配下标

    def swap_sub_sup(match):
        # 这是一个简化处理，假设我们想要固定为 "先下标后上标" (_ first, ^ second)
        # 如果捕获到的是 ^..._...，则交换
        full_str = match.group(0)
        if full_str.startswith("^"):
            # 找到分割点并不容易，这里建议使用简单的启发式或多次 pass
            # 简单策略：如果存在 ^ 和 _，且 ^ 在 _ 前面，我们尝试交换它们
            # 为保证鲁棒性，推荐使用专门的 Tokenizer，但在正则层面上：
            parts = full_str.split("_")
            if len(parts) == 2:
                return "_" + parts[1] + parts[0]
        return full_str

    # 简单的正则无法完美处理嵌套花括号，
    # 但对于标准化评估，通常我们会运行多遍或者假定生成结果结构较简单。
    # 这里提供一个针对常见 x^a_b 格式的替换:
    # 匹配: ^(组)_(组) -> _(组)^(组)
    # 组可以是 {xxx} 或 单字符
    pattern_sup_sub = (
        r"(\^(?:\{[^{}]+\}|[a-zA-Z0-9\\]+))(_(?:\{[^{}]+\}|[a-zA-Z0-9\\]+))"
    )

    # 循环替换直到没有变化（处理连续的情况）
    while True:
        new_s = re.sub(pattern_sup_sub, lambda m: m.group(2) + m.group(1), s)
        if new_s == s:
            break
        s = new_s

    # 8. 花括号标准化 (可选)
    # 策略 A: 移除单字符的花括号 x^{2} -> x^2
    s = re.sub(r"([_^])\{([a-zA-Z0-9])\}", r"\1\2", s)
    # 策略 C: 将 ^{\\infty} -> ^\infty, _{\\alpha} -> _\alpha 等（去掉包裹命令的花括号）
    s = re.sub(r"\^\{(\\[a-zA-Z]+)\}", r"^\1", s)
    s = re.sub(r"_\{(\\[a-zA-Z]+)\}", r"_\1", s)
    # 移除围绕单个 \\frac 的多余括号 (例如由 \left(\frac{...}\right) 产生)
    s = re.sub(r"\((\\frac\{[^}]+\}\{[^}]+\})\)", r"\1", s)

    # 策略 B: (如果你想保留所有花括号) x^2 -> x^{2}
    # s = re.sub(r'([_^])([a-zA-Z0-9])', r'\1{\2}', s)

    return s


def similarity(pred: str, gt: str):
    edit_distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    normalized_distance = edit_distance / max_len if max_len > 0 else 1
    return 1.0 - normalized_distance


def math_formula_eds_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        # normalize latex
        solution_str = normalize_latex(solution_str)
        ground_truth = normalize_latex(ground_truth)

        reward = similarity(solution_str, ground_truth)
        return reward
    except Exception as e:
        traceback.print_exc()
        return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = None,
    extra_info: Union[str, dict] = None,
    format_score: float = 0.3,
) -> dict:
    accuracy_score = math_formula_eds_reward(solution_str, ground_truth)
    result = {
        "score": accuracy_score,
        "format": 0,
        "accuracy": accuracy_score,
    }
    return result


if __name__ == "__main__":
    # 测试用例
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
        pd_out = normalize_latex(pd_inp)
        gt_out = normalize_latex(gt_inp)
        status = "PASS" if pd_out == gt_out else "!!!FAIL!!!"
        print(f"====== case {idx} =====")
        print(f"pd_inp: {pd_inp}")
        print(f"gt_inp: {gt_inp}")
        print(f"pd_out: {pd_out}")
        print(f"gt_out: {gt_out}")
        print(f"status: {status}")
