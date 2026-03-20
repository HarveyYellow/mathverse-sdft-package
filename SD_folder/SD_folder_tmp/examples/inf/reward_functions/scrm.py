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
from typing import List, Dict, Tuple
import json
import Levenshtein
import numpy as np
import traceback


def extract_valid_content(text):

    patterns = [
        r"```json\n(.*?)\n```",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def csv_eval(predictions, references, easy, pred_type="json"):
    """CSV 结构评测函数，计算 mPrecision"""

    def is_int(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def is_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    def convert_dict_to_list(data):
        """将字典转换为三元组列表"""
        converted_list = []
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    converted_list.append(
                        (key, subkey, re.sub(r"[^\d.-]", "", str(subvalue)))
                    )
            else:
                converted_list.append(
                    (key, "value", re.sub(r"[^\d.-]", "", str(value)))
                )
        return converted_list

    def process_triplets(triplets):
        """处理三元组，转换数据类型"""
        new_triplets = []
        for triplet in triplets:
            if len(triplet) > 2:
                if is_int(triplet[2]) or is_float(triplet[2]):
                    triplet_temp = (
                        triplet[0].lower(),
                        triplet[1].lower(),
                        float(triplet[2]),
                    )
                else:
                    triplet_temp = (
                        triplet[0].lower(),
                        triplet[1].lower(),
                        triplet[2].lower(),
                    )
            else:
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
            new_triplets.append(triplet_temp)
        return new_triplets

    def intersection_with_tolerance(a, b, tol_word, tol_num):
        """计算带容差的交集"""
        a = set(a)
        b = set(b)
        c = set()
        for elem1 in a:
            for elem2 in b:
                if is_float(elem1[-1]) and is_float(elem2[-1]):
                    if (
                        (
                            (
                                Levenshtein.distance(
                                    "".join(elem1[:-1]), "".join(elem2[:-1])
                                )
                                <= tol_word
                            )
                            and (
                                abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1]) + 0.000001)
                                <= tol_num
                            )
                        )
                        or (
                            ("".join(elem1[:-1]) in "".join(elem2[:-1]))
                            and (
                                abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1]) + 0.000001)
                                <= tol_num
                            )
                        )
                        or (
                            ("".join(elem2[:-1]) in "".join(elem1[:-1]))
                            and (
                                abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1]) + 0.000001)
                                <= tol_num
                            )
                        )
                    ):
                        c.add(elem1)
                else:
                    if (
                        Levenshtein.distance(
                            "".join([str(i) for i in elem1]),
                            "".join([str(j) for j in elem2]),
                        )
                        <= tol_word
                    ):
                        c.add(elem1)
        return list(c)

    def union_with_tolerance(a, b, tol_word, tol_num):
        """计算带容差的并集"""
        c = set(a) | set(b)
        d = set(a) & set(b)
        e = intersection_with_tolerance(a, b, tol_word, tol_num)
        f = set(e)
        g = c - (f - d)
        return list(g)

    def get_eval_list(pred_csv, label_csv, tol_word, tol_num):
        """获取评测列表"""
        pred_triple_list = []
        for it in pred_csv:
            pred_triple_temp = convert_dict_to_list(it)
            pred_triple_pre = process_triplets(pred_triple_temp)
            pred_triple_list.append(pred_triple_pre)

        label_triple_list = []
        for it in label_csv:
            label_triple_temp = convert_dict_to_list(it)
            label_triple_pre = process_triplets(label_triple_temp)
            label_triple_list.append(label_triple_pre)

        intersection_list = []
        union_list = []
        sim_list = []

        for pred, label in zip(pred_triple_list, label_triple_list):
            for idx in range(len(pred)):
                try:
                    if label[idx][1] == "value" and "value" not in pred[idx][:2]:
                        pred[idx] = (pred[idx][0], "value", pred[idx][2])
                    temp_pred_head = sorted(pred[idx][:2])
                    temp_gt_head = sorted(label[idx][:2])
                    pred[idx] = (temp_pred_head[0], temp_pred_head[1], pred[idx][2])
                    label[idx] = (temp_gt_head[0], temp_gt_head[1], label[idx][2])
                except:
                    continue
            intersection = intersection_with_tolerance(
                pred, label, tol_word=tol_word, tol_num=tol_num
            )
            union = union_with_tolerance(
                pred, label, tol_word=tol_word, tol_num=tol_num
            )
            sim = len(intersection) / len(union) if len(union) > 0 else 0.0
            intersection_list.append(intersection)
            union_list.append(union)
            sim_list.append(sim)
        return intersection_list, union_list, sim_list

    def get_ap(predictions, labels, sim_threshold, tolerance, easy):
        """计算 Average Precision"""
        if tolerance == "strict":
            tol_word = 0
            tol_num = 0 if easy == 1 else 0.1
        elif tolerance == "slight":
            tol_word = 2
            tol_num = 0.05 if easy == 1 else 0.3
        elif tolerance == "high":
            tol_word = 5
            tol_num = 0.1 if easy == 1 else 0.5

        _, _, sim_list = get_eval_list(
            predictions, labels, tol_word=tol_word, tol_num=tol_num
        )
        ap = len([num for num in sim_list if num >= sim_threshold]) / (
            len(sim_list) + 1e-16
        )
        return ap

    # 主评测逻辑
    map_strict = 0
    map_slight = 0
    map_high = 0

    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += (
            get_ap(predictions, references, sim_threshold, "strict", easy) / 10
        )
        map_slight += (
            get_ap(predictions, references, sim_threshold, "slight", easy) / 10
        )
        map_high += get_ap(predictions, references, sim_threshold, "high", easy) / 10

    # em = get_ap(predictions, references, 1, 'strict', easy)
    # ap_50_strict = get_ap(predictions, references, 0.5, 'strict', easy)
    # ap_75_strict = get_ap(predictions, references, 0.75, 'strict', easy)
    # ap_90_strict = get_ap(predictions, references, 0.90, 'strict', easy)
    # ap_50_slight = get_ap(predictions, references, 0.5, 'slight', easy)
    # ap_75_slight = get_ap(predictions, references, 0.75, 'slight', easy)
    # ap_90_slight = get_ap(predictions, references, 0.90, 'slight', easy)
    # ap_50_high = get_ap(predictions, references, 0.5, 'high', easy)
    # ap_75_high = get_ap(predictions, references, 0.75, 'high', easy)
    # ap_90_high = get_ap(predictions, references, 0.90, 'high', easy)

    return (map_strict + map_slight + map_high) / 3.0


def scrm_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        # load json
        pred_json = json.loads(solution_str)
        gt_json = json.loads(ground_truth)

        # 收集 mPrecision 需要的数据
        predictions = [pred_json.get("values", {})]
        labels = [gt_json["values"]]

        # # 收集 OCR/EM 需要的数据
        # title_preds = [pred_json.get('title', 'None')]
        # title_gts = [gt_json['gts'].get('title', 'None')]

        # source_preds = [pred_json.get('source', 'None')]
        # source_gts = [gt_json['gts'].get('source', 'None')]

        # x_title_preds = [pred_json.get('x_title', 'None')]
        # x_title_gts = [gt_json['gts'].get('x_title', 'None')]

        # y_title_preds = [pred_json.get('y_title', 'None')]
        # y_title_gts = [gt_json['gts'].get('y_title', 'None')]

        map_score = csv_eval(
            predictions=predictions,
            references=labels,
            easy=1,  # ChartX 使用 easy=1
            pred_type="json",
        )

        # title_em = ocr_eval(title_gts, title_preds)
        # source_em = ocr_eval(source_gts, source_preds)
        # x_title_em = ocr_eval(x_title_gts, x_title_preds)
        # y_title_em = ocr_eval(y_title_gts, y_title_preds)

        reward = map_score
        return reward
    except Exception as e:
        traceback.print_exc()
        return 0.0
