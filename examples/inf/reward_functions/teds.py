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

import markdown
import re
import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm
import numpy as np
import traceback


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS(object):
    """Tree Edit Distance basead Similarity"""

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1
        ), "n_jobs must be an integer greather than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        if (not pred) or (not true):
            return 0.0
        if "<body>" not in pred:
            pred = "<html><body>{}</body></html>".format(pred)
        if "<body>" not in true:
            true = "<html><body>{}</body></html>".format(true)

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        """Computes TEDS score between the prediction and the ground truth of
        a batch of samples
        @params pred_json: {'FILENAME': 'HTML CODE', ...}
        @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
        @output: {'FILENAME': 'TEDS SCORE', ...}
        """
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [
                self.evaluate(pred_json.get(filename, ""), true_json[filename]["html"])
                for filename in tqdm(samples)
            ]
        else:
            inputs = [
                {
                    "pred": pred_json.get(filename, ""),
                    "true": true_json[filename]["html"],
                }
                for filename in samples
            ]
            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1
            )
        scores = dict(zip(samples, scores))
        return scores


def extract_valid_content(text):

    patterns = [
        r"```markdown\n(.*?)\n```",
        r"```html\n(.*?)\n```",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def replace_inner_newlines_with_space(md: str) -> str:
    """
    替换 markdown 表格 **单元格中的换行符** 为空格，
    保留每一行之间的 \n，使表格结构不受破坏。
    """
    if "|\n|" not in md:
        return md
    md = md.strip()
    lines = md.split("|\n|")
    fixed_lines = []

    for line in lines:
        line = line.replace("\n", " ")
        fixed_lines.append(line)

    return '|\n|'.join(fixed_lines)


def convert_markdown2html(markdown_str):
    extensions = ["markdown.extensions.tables"]
    html_str = markdown.markdown(replace_inner_newlines_with_space(markdown_str), extensions=extensions)
    if "<html><body>" not in html_str:
        html_str = "<html><body>{}</body></html>".format(html_str)
    return html_str


def normalize_head_body(html_str: str) -> str:
    """
    Remove <thead> and <tbody> tags and convert <th> to <td> in the given HTML string.
    Preserves attributes on <th> when converting to <td>.
    """
    if not html_str:
        return html_str
    # remove thead and tbody tags (opening and closing, with any attributes)
    html_str = re.sub(r"</?thead[^>]*>", "", html_str, flags=re.IGNORECASE | re.DOTALL)
    html_str = re.sub(r"</?tbody[^>]*>", "", html_str, flags=re.IGNORECASE | re.DOTALL)
    # convert opening th tags to td, preserving attributes
    html_str = re.sub(r"<\s*th([^>]*)>", r"<td\1>", html_str, flags=re.IGNORECASE)
    # convert closing th tags to td
    html_str = re.sub(r"</\s*th\s*>", "</td>", html_str, flags=re.IGNORECASE)
    return html_str


def teds_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        # convert md table to html table
        if "<table>" not in solution_str:
            solution_str = convert_markdown2html(solution_str)
        if "<table>" not in ground_truth:
            ground_truth = convert_markdown2html(ground_truth)

        # normalize table head/body and th cells
        solution_str = normalize_head_body(solution_str)
        ground_truth = normalize_head_body(ground_truth)

        teds_obj = TEDS(structure_only=False)
        reward = teds_obj.evaluate(solution_str, ground_truth)
        return reward
    except Exception as e:
        traceback.print_exc()
        return 0.0
