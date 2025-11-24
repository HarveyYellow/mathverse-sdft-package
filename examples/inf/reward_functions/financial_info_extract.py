import copy
import json
import re
import math
from typing import Union, Optional, List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor


class NumericProcessor:
    """数值处理工具类"""

    @staticmethod
    def extract_digits(text: Union[str, int, float]) -> str:
        """提取字符串中最左侧和最右侧数字之间的所有数字字符"""
        if isinstance(text, (int, float)):
            text = str(text)

        digit_matches = list(re.finditer(r"\d", text))
        if not digit_matches:
            return ""

        start_index = digit_matches[0].start()
        end_index = digit_matches[-1].end()
        substring = text[start_index:end_index]
        digits_only = re.sub(r"\D", "", substring)
        return digits_only

    @staticmethod
    def parse_float_value(
        value: Union[str, int, float], unit: Optional[str] = None
    ) -> Optional[float]:
        """解析字符串value和对应单位unit，将其转换为float数值并进行单位换算"""
        unit_map = {
            "thousand": 1e3,
            "ten thousand": 1e4,
            "hundred thousand": 1e5,
            "million": 1e6,
            "ten million": 1e7,
            "hundred million": 1e8,
            "billion": 1e9,
            "ten billion": 1e10,
            "hundred billion": 1e11,
            "trillion": 1e12,
            "percent": 0.01,
            "percentage": 0.01,
            "千": 1e3,
            "万": 1e4,
            "十万": 1e5,
            "百万": 1e6,
            "千万": 1e7,
            "亿": 1e8,
            "十亿": 1e9,
            "百亿": 1e10,
            "万亿": 1e12,
            "百分比": 0.01,
            "unknown": 1.0,
        }

        def process_unit(unit_str):
            if unit_str is None:
                return 1.0
            if not isinstance(unit_str, str):
                unit_str = str(unit_str)
            unit_str = unit_str.strip().lower()
            for k, v in unit_map.items():
                if unit_str == k or unit_str.endswith(k):
                    return v
            return 1.0

        if isinstance(value, (int, float)):
            return float(value) * process_unit(unit)

        if not isinstance(value, str):
            return None

        value = value.strip()
        if not value:
            return None

        first_digit_match = re.search(r"[0-9]", value)
        if not first_digit_match:
            return None

        first_digit_pos = first_digit_match.start()
        last_digit_match = re.search(r"[0-9][^0-9]*$", value)
        last_digit_pos = (
            last_digit_match.start() if last_digit_match else len(value) - 1
        )

        prefix = value[:first_digit_pos].strip()
        suffix = value[last_digit_pos + 1 :].strip()
        first_non_space_char = prefix[-1] if prefix else None
        last_non_space_char = suffix[0] if suffix else None

        num_region = value[first_digit_pos : last_digit_pos + 1]
        is_negative = first_non_space_char in ["(", "-"] or last_non_space_char == ")"

        last_dot_index = num_region.rfind(".")
        digits = []
        for i, char in enumerate(num_region):
            if char.isdigit() or (char == "." and i == last_dot_index):
                digits.append(char)

        clean_num_str = "".join(digits)
        try:
            num = float(clean_num_str)
        except Exception:
            return None

        return (-abs(num) if is_negative else num) * process_unit(unit)


class MetricsCalculator:
    """指标计算核心类"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        #print(f"self.config: {self.config}")
        if self.config.get("data_type", "in_domain") == "in_domain":
            self.attr_func_dict = {}
        else:
            self.attr_func_dict = {
                "float_value": self.get_norm_float_value,
                "unit": self.get_norm_unit_value,
            }

    def get_norm_float_value(self, input_obj: Dict) -> str:
        """获取标准化数值"""
        value = input_obj.get("value", "")
        unit = input_obj.get("unit", "")
        if value == "unknown":
            return "unknown"

        float_value = NumericProcessor.parse_float_value(value, unit)
        return float_value

    def get_norm_unit_value(self, input_obj: Dict) -> str:
        """获取标准化单位值"""
        value = str(input_obj.get("unit", None))
        if value in ["", "unknown", "None"]:
            return "unknown"
        return value

    def get_norm_attr_value(self, input_obj: Dict, attr_name: str) -> str:
        """获取标准化属性值"""
        default_func = lambda x: str(x.get(attr_name, None))
        field_func = self.attr_func_dict.get(attr_name, default_func)
        field_value = field_func(input_obj)
        return field_value

    def match_metrics(
        self, gt_obj: Dict, pd_obj: Dict, gt_attr_name: str, pd_attr_name: str
    ) -> Tuple[List, List, List, List]:
        """匹配指标计算"""
        g_tp_list = []
        g_tn_list = []
        g_fp_list = []
        g_fn_list = []

        gt_valid = gt_obj is not None and len(gt_obj) > 0
        pd_valid = pd_obj is not None and len(pd_obj) > 0

        if not gt_valid and not pd_valid:
            raise ValueError("input match_overall_metrics error!!!")

        # 从config获取attr_names列表
        if gt_attr_name == "overall" and pd_attr_name == "overall":
            attr_names = self.config.get(
                "attr_names", ["value", "unit", "currency", "cutoff_date"]
            )
        else:
            attr_names = [gt_attr_name]

        def update_metrics(
            field_name,
            attr_name,
            gt_obj,
            gold_value,
            pd_obj,
            pred_value,
            tp,
            tn,
            fp,
            fn,
        ):
            if gold_value == pred_value:
                if gold_value == "unknown":
                    tn.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                else:
                    tp.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
            else:
                if gold_value != "unknown" and pred_value == "unknown":
                    fn.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                if gold_value == "unknown" and pred_value != "unknown":
                    fp.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                if gold_value != "unknown" and pred_value != "unknown":
                    fn.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                    fp.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
            return tp, tn, fp, fn

        if gt_valid and pd_valid:
            field_name = gt_obj.get("field_name", "unknown")
            for attr_name in attr_names:
                gt_value = self.get_norm_attr_value(gt_obj, attr_name)
                pd_value = self.get_norm_attr_value(pd_obj, attr_name)
                g_tp_list, g_tn_list, g_fp_list, g_fn_list = update_metrics(
                    field_name,
                    attr_name,
                    gt_obj,
                    gt_value,
                    pd_obj,
                    pd_value,
                    g_tp_list,
                    g_tn_list,
                    g_fp_list,
                    g_fn_list,
                )

        if gt_valid and not pd_valid:
            field_name = gt_obj.get("field_name", "unknown")
            for attr_name in attr_names:
                gt_value = self.get_norm_attr_value(gt_obj, attr_name)
                if gt_value == "unknown":
                    g_tn_list.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                else:
                    g_fn_list.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )

        if not gt_valid and pd_valid:
            field_name = pd_obj.get("field_name", "unknown")
            for attr_name in attr_names:
                pd_value = self.get_norm_attr_value(pd_obj, attr_name)
                if pd_value == "unknown":
                    g_tn_list.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )
                else:
                    g_fp_list.append(
                        {
                            "field_name": field_name,
                            "attr_name": attr_name,
                            "gt_obj": gt_obj,
                            "pd_obj": pd_obj,
                        }
                    )

        # 验证计算结果
        pred_not_unknown = 0
        if pd_valid:
            for attr_name in attr_names:
                pd_value = self.get_norm_attr_value(pd_obj, attr_name)
                if pd_value != "unknown":
                    pred_not_unknown += 1

        gold_not_unknown = 0
        if gt_valid:
            for attr_name in attr_names:
                gt_value = self.get_norm_attr_value(gt_obj, attr_name)
                if gt_value != "unknown":
                    gold_not_unknown += 1

        assert len(g_tp_list) + len(g_tn_list) + len(g_fp_list) + len(g_fn_list) > 0
        assert len(g_tp_list) + len(g_fp_list) == pred_not_unknown
        assert len(g_tp_list) + len(g_fn_list) == gold_not_unknown

        return g_tp_list, g_tn_list, g_fp_list, g_fn_list

    def example_match_predictions(
        self,
        gt_objs: List[Dict],
        pd_objs: List[Dict],
        gt_attr_name: str,
        pd_attr_name: str,
    ) -> Tuple[List, List, List, List]:
        """匹配一个样本内的目标，返回TP,FP,TN,FN的详情列表"""
        matched_gt = set()
        matched_pd = set()
        final_tp_list = []
        final_tn_list = []
        final_fp_list = []
        final_fn_list = []

        # 过滤用于召回的recall的内容
        pred_objs = [pred for pred in pd_objs if pred.get("phrase", "") != "recall"]

        # 处理正样本
        for gt_idx, gt in enumerate(gt_objs):
            gt_cat = gt.get("field_name", None)

            if gt_idx in matched_gt:
                continue

            best_pd_match_id = -1
            for pred_idx, pred in enumerate(pred_objs):
                pred_cat = pred.get("field_name", None)
                if gt_cat != pred_cat or pred_idx in matched_pd:
                    continue
                best_pd_match_id = pred_idx
                break

            if best_pd_match_id != -1:
                matched_gt.add(gt_idx)
                matched_pd.add(best_pd_match_id)
                pd = pred_objs[best_pd_match_id]
                tp_list, tn_list, fp_list, fn_list = self.match_metrics(
                    gt, pd, gt_attr_name, pd_attr_name
                )
            else:
                tp_list, tn_list, fp_list, fn_list = self.match_metrics(
                    gt, {}, gt_attr_name, pd_attr_name
                )

            final_tp_list.extend(tp_list)
            final_tn_list.extend(tn_list)
            final_fp_list.extend(fp_list)
            final_fn_list.extend(fn_list)

        # 处理未匹配的预测对象
        for pred_idx, pred in enumerate(pred_objs):
            if pred_idx not in matched_pd:
                tp_list, tn_list, fp_list, fn_list = self.match_metrics(
                    {}, pred, gt_attr_name, pd_attr_name
                )
                final_tp_list.extend(tp_list)
                final_tn_list.extend(tn_list)
                final_fp_list.extend(fp_list)
                final_fn_list.extend(fn_list)

        return final_tp_list, final_tn_list, final_fp_list, final_fn_list

    def compute_precision_recall(
        self,
        gt_list: List[List[Dict]],
        pd_list: List[List[Dict]],
        data_info_list: List[Dict],
        gt_attr_name: str,
        pd_attr_name: str,
    ) -> List[Dict]:
        """计算精度和召回率，支持多样本"""
        assert len(gt_list) == len(pd_list), "输入图像数量不一致"

        final_data_result = []
        for example_gt_objs, example_pd_objs, data_info in zip(
            gt_list, pd_list, data_info_list
        ):
            tp_list, tn_list, fp_list, fn_list = self.example_match_predictions(
                example_gt_objs, example_pd_objs, gt_attr_name, pd_attr_name
            )

            TP = len(tp_list)
            FP = len(fp_list)
            FN = len(fn_list)
            TN = len(tn_list)

            data_info["TP_LIST"] = tp_list
            data_info["TN_LIST"] = tn_list
            data_info["FP_LIST"] = fp_list
            data_info["FN_LIST"] = fn_list

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0

            if precision == 0 and recall == 0:
                if TP == 0 and FP == 0 and FN == 0 and TN > 0:
                    precision = 1.0
                    recall = 1.0

            pr_sum = precision + recall
            f1 = 0.0 if pr_sum == 0 else 2 * precision * recall / pr_sum

            data_info["overall"] = {"precision": precision, "recall": recall, "f1": f1}
            final_data_result.append(data_info)

        return final_data_result


def extract_data(response):
    response = json.loads(response.split("``json")[-1].split("```")[0].strip())
    return response


def compute_score(solution_str, ground_truth, **kwargs):
    calculator = MetricsCalculator({"attr_names": ["value", "unit", "currency", "cutoff_date"]})
    ground_truth = extract_data(ground_truth)
    try:
        solution = extract_data(solution_str)
        data_info = {"id": 1}
        result = calculator.compute_precision_recall([ground_truth], [solution], [data_info], "overall", "overall")[0]
        reward = result["overall"]["f1"]
        format_reward = 1
    except:
        reward = 0
        format_reward = 0
    return {"score": reward + format_reward, "reward": reward, "format_reward": format_reward}
