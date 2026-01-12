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
from typing import List, Dict, Tuple, Optional
import Levenshtein
import dataclasses
import numpy as np
from scipy import optimize
import itertools


def extract_valid_content(text):

    patterns = [
        r"```markdown\n(.*?)\n```",
        r"```html\n(.*?)\n```",
        r"```latex\n(.*?)\n```",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def _anls_metric(s1, s2, theta=0.5):
    """Computes average normalized levenshtein similarity (ANLS)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    distance = Levenshtein.distance(s1, s2)
    anls = 1 - distance / max(len(s1), len(s2))
    return anls if anls >= theta else 0.0


def _permute(values, indexes):
    """Helper to permute a tuple."""
    return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
    """Helper class for the content of a markdown table."""

    title: Optional[str] = None
    headers: tuple = dataclasses.field(default_factory=tuple)
    rows: tuple = dataclasses.field(default_factory=tuple)

    def permuted(self, indexes):
        """Builds a version of the table changing the column order."""
        return Table(
            title=self.title,
            headers=_permute(self.headers, indexes),
            rows=tuple(_permute(row, indexes) for row in self.rows),
        )

    def aligned(self, headers, text_theta=0.5):
        """Builds a column permutation with headers in the most correct order."""
        if len(headers) != len(self.headers):
            raise ValueError(f"Header length {headers} must match {self.headers}.")
        distance = []
        for h2 in self.headers:
            distance.append([1 - _anls_metric(h1, h2, text_theta) for h1 in headers])
        cost_matrix = np.array(distance)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
        score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
        return self.permuted(permutation), score


def _parse_table(text, transposed=False):
    """Builds a table from a markdown representation."""
    lines = text.lower().splitlines()
    if not lines:
        return Table()

    title = None
    offset = 0
    first_line_parts = [part.strip() for part in lines[0].strip(" |").split("|")]
    if first_line_parts[0] == "title":
        title = first_line_parts[1] if len(first_line_parts) > 1 else ""
        offset = 1
    elif lines[0].startswith("title |"):
        title = lines[0][len("title |") :].strip()
        offset = 1

    if len(lines) < offset:
        return Table(title=title)

    rows = []
    for line in lines[offset:]:
        if not line.strip():
            continue
        if all(c in "|- " for c in line):
            continue
        row_parts = [v.strip() for v in line.strip(" |").split("|")]
        rows.append(tuple(row_parts))

    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]

    if not rows:
        return Table(title=title)

    return Table(title=title, headers=rows[0], rows=tuple(rows[1:]))


def _get_table_datapoints(table: Table):
    """Extracts a dict of datapoints from a table."""
    datapoints: Dict[str, str] = {}
    if table.title is not None:
        datapoints["title"] = table.title
    if not table.rows or len(table.headers) <= 1:
        return datapoints
    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell
    return datapoints


def _to_float(text):
    """Convert text to float, handling percentages."""
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        else:
            return float(text)
    except ValueError:
        return None


def _get_relative_distance(target, prediction, theta=1.0):
    """Returns min(1, |target-prediction|/|target|)."""
    if not target:
        return int(not prediction)
    distance = min(abs((target - prediction) / target), 1)
    return distance if distance < theta else 1


def _get_datapoint_metric(
    target: Tuple[str, str],
    prediction: Tuple[str, str],
    text_theta: float = 0.5,
    number_theta: float = 0.1,
):
    """Computes a metric that scores how similar two datapoint pairs are."""
    key_metric = _anls_metric(target[0], prediction[0], text_theta)
    pred_float = _to_float(prediction[1])
    target_float = _to_float(target[1])
    if pred_float is not None and target_float:
        return key_metric * (
            1 - _get_relative_distance(target_float, pred_float, number_theta)
        )
    elif target[1] == prediction[1]:
        return key_metric
    else:
        return key_metric * _anls_metric(target[1], prediction[1], text_theta)


def _table_datapoints_precision_recall_f1(
    target_table: Table,
    prediction_table: Table,
    text_theta: float = 0.5,
    number_theta: float = 0.1,
):
    """Datapoint-level precision/recall/F1 between two tables."""
    target_datapoints = list(_get_table_datapoints(target_table).items())
    prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
    if not target_datapoints and not prediction_datapoints:
        return 1.0, 1.0, 1.0
    if not target_datapoints:
        return 0.0, 1.0, 0.0
    if not prediction_datapoints:
        return 1.0, 0.0, 0.0

    distance = []
    for t, _ in target_datapoints:
        distance.append(
            [1 - _anls_metric(t, p, text_theta) for p, _ in prediction_datapoints]
        )
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    score = 0.0
    for r, c in zip(row_ind, col_ind):
        score += _get_datapoint_metric(
            target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
        )
    if score == 0:
        return 0.0, 0.0, 0.0
    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    return precision, recall, 2 * precision * recall / (precision + recall)


def rmsf1_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        # calculate reward
        pred = solution_str
        target_list = [ground_truth]
        all_metrics = []
        for transposed in [True, False]:
            pred_table = _parse_table(pred, transposed=transposed)
            all_metrics.extend(
                [
                    _table_datapoints_precision_recall_f1(
                        _parse_table(t), pred_table, text_theta, number_theta
                    )
                    for t in target_list
                ]
            )
        p, r, f = max(all_metrics, key=lambda x: x[-1])
        reward = f

        return reward
    except Exception as e:
        return 0.0
