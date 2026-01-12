import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import Levenshtein
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def extract_json_content(text):
    matches = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if matches:
        text = matches.group(1).strip()
    else:
        partial_md_match = re.search(r"```json\n(.*)", text, re.DOTALL)
        if partial_md_match:
            text = partial_md_match.group(1).strip()
    return text


def truncate_last_incomplete_element(text: str):
    """Truncates the last incomplete element"""

    # For very long text (>50k) or text not ending with ']', directly truncate the last '{"bbox":'
    needs_truncation = len(text) > 50000 or not text.strip().endswith("]")

    if needs_truncation:
        # Check how many dict objects there are
        bbox_count = text.count('{"bbox":')

        # If there is only one dict object, do not truncate to avoid deleting the only object
        if bbox_count <= 1:
            # print(f"    ⚠️ Only {bbox_count} dict objects found, skipping truncation to avoid deleting all content")
            return text, False

        # Find the position of the last '{"bbox":'
        last_bbox_pos = text.rfind('{"bbox":')

        if last_bbox_pos > 0:
            # Truncate before this position
            truncated_text = text[:last_bbox_pos].rstrip()

            # Remove trailing comma
            if truncated_text.endswith(","):
                truncated_text = truncated_text[:-1]
                truncated_text = truncated_text + "]"
            # print(f"    ✂️ Truncated the last incomplete element, length reduced from {len(text):,} to {len(truncated_text):,}")
            return truncated_text, True

    return text, False


class Doc2JsonReward:
    def __init__(self, text_weight: float = 0.5, layout_weight: float = 0.5):
        """
        初始化奖励函数

        Args:
            text_weight: 文本相似度奖励的权重
            layout_weight: 布局奖励的权重
        """
        self.text_weight = text_weight
        self.layout_weight = layout_weight

    def _normalize_items(self, items: List[Dict]) -> List[Dict]:
        """Ensure every item has a 'text' key (fill with empty string if missing)."""
        normalized = []
        for item in items:
            it = dict(item)
            if "text" not in it:
                it["text"] = ""
            normalized.append(it)
        return normalized

    def calculate_text_similarity_reward(
        self, pred: List[Dict], gt: List[Dict]
    ) -> float:
        """
        计算文本相似度奖励

        Args:
            pred: 模型预测数据，格式同gt
            gt: ground truth数据，格式为[{"bbox": [...], "category": "...", "text": "..."}, ...]

        Returns:
            文本相似度奖励值
        """
        # 按顺序用"\n\n"连接所有text内容
        pred_text = "\n\n".join([item["text"] for item in pred if "text" in item])
        gt_text = "\n\n".join([item["text"] for item in gt if "text" in item])

        # 计算编辑距离
        edit_distance = Levenshtein.distance(pred_text, gt_text)

        # 计算归一化编辑距离
        max_length = max(len(pred_text), len(gt_text))
        if max_length == 0:
            normalized_edit_distance = 0.0
        else:
            normalized_edit_distance = edit_distance / max_length

        # 奖励为 1 - normalized_edit_distance
        text_reward = 1.0 - normalized_edit_distance

        return max(0.0, min(1.0, text_reward))  # 确保在[0,1]范围内

    def bbox_to_polygon(self, bbox: List[float]) -> Polygon:
        """将bbox转换为shapely多边形"""
        x1, y1, x2, y2 = bbox
        return box(x1, y1, x2, y2)

    def create_category_mask(self, items: List[Dict], category: str) -> float:
        """
        为指定类别创建mask区域（使用shapely计算多边形并集）

        Args:
            items: 包含bbox的数据项
            category: 目标类别

        Returns:
            该类别所有bbox并集的面积
        """
        # 获取该类别所有bbox
        bboxes = [item["bbox"] for item in items if item["category"] == category]

        if not bboxes:
            return 0.0

        # 将所有bbox转换为多边形
        polygons = [self.bbox_to_polygon(bbox) for bbox in bboxes]

        # 计算所有多边形的并集
        union_polygon = unary_union(polygons)

        # 返回并集面积
        return union_polygon.area

    def calculate_category_iou(
        self, pred_items: List[Dict], gt_items: List[Dict], category: str
    ) -> float:
        """
        计算指定类别的IoU（使用多边形并集方法）

        Args:
            pred_items: 预测数据
            gt_items: ground truth数据
            category: 目标类别

        Returns:
            该类别的IoU值
        """
        # 获取该类别在pred和gt中的所有bbox
        pred_bboxes = [
            item["bbox"] for item in pred_items if item["category"] == category
        ]
        gt_bboxes = [item["bbox"] for item in gt_items if item["category"] == category]

        # 如果都没有该类别，返回1.0
        if not pred_bboxes and not gt_bboxes:
            return 1.0

        # 如果只有一方有该类别，返回0.0
        if not pred_bboxes or not gt_bboxes:
            return 0.0

        # 将pred中该类别的所有bbox转换为多边形并计算并集
        pred_polygons = [self.bbox_to_polygon(bbox) for bbox in pred_bboxes]
        pred_union = unary_union(pred_polygons)
        # print(f"pred_area: {pred_union.area}")

        # 将gt中该类别的所有bbox转换为多边形并计算并集
        gt_polygons = [self.bbox_to_polygon(bbox) for bbox in gt_bboxes]
        gt_union = unary_union(gt_polygons)
        # print(f"gt_area: {gt_union.area}")

        # 计算交集面积
        intersection_area = pred_union.intersection(gt_union).area
        # print(f"intersection area: {intersection_area}")

        # 计算并集面积
        union_area = pred_union.union(gt_union).area
        # print(f"union area: {union_area}")
        # 计算IoU
        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def calculate_layout_reward(self, pred: List[Dict], gt: List[Dict]) -> float:
        """
        计算布局奖励（mIOU）

        Args:
            pred: 模型预测数据
            gt: ground truth数据

        Returns:
            布局奖励值
        """
        # 获取所有类别
        pred_categories = set(item["category"] for item in pred)
        gt_categories = set(item["category"] for item in gt)
        all_categories = pred_categories.union(gt_categories)

        if not all_categories:
            return 1.0  # 如果没有类别，认为是完美的

        category_iou_scores = []

        for category in all_categories:
            # 计算该类别的IoU
            category_iou = self.calculate_category_iou(pred, gt, category)
            category_iou_scores.append(category_iou)

        # 计算平均IoU（mIOU）
        mean_iou = np.mean(category_iou_scores) if category_iou_scores else 0.0

        return mean_iou

    def __call__(self, pred_str: str, gt_str: str) -> float:
        """
        计算总奖励

        Args:
            pred_str: 模型预测数据
            gt_str: ground truth数据

        Returns:
            total_reward: 总奖励值
        """
        pred_json = extract_json_content(pred_str)
        pred_json, trunc = truncate_last_incomplete_element(pred_json)
        gt_json = extract_json_content(gt_str)

        try:
            pred = json.loads(pred_json)
            gt = json.loads(gt_json)
            # 支持两种格式：带或不带"text"字段，统一填充为空字符串以便后续处理
            pred = self._normalize_items(pred)
            gt = self._normalize_items(gt)
            # 计算文本相似度奖励
            text_reward = self.calculate_text_similarity_reward(pred, gt)

            # 计算布局奖励
            layout_reward = self.calculate_layout_reward(pred, gt)

            # 计算加权奖励
            total_reward = (
                self.text_weight * text_reward + self.layout_weight * layout_reward
            )
            return (total_reward, text_reward, layout_reward), trunc, None
        except Exception as e:
            return (0, 0, 0), trunc, (e, pred_json)


# 使用示例
if __name__ == "__main__":
    # 示例数据
    pred_data = [
        {"bbox": [12, 12, 98, 32], "category": "title", "text": "文档标题"},
        {
            "bbox": [15, 42, 195, 105],
            "category": "paragraph",
            "text": "这是一个段落内容",
        },
        {"bbox": [8, 108, 305, 205], "category": "table", "text": "表格数据"},
        {"bbox": [45, 208, 255, 305], "category": "table", "text": "另一个表格"},
    ]
    gt_data = [
        {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
        {
            "bbox": [10, 40, 200, 100],
            "category": "paragraph",
            "text": "这是一个段落内容",
        },
        {"bbox": [10, 110, 300, 200], "category": "table", "text": "表格数据"},
        {"bbox": [50, 210, 250, 300], "category": "table", "text": "另一个表格"},
    ]

    # 创建奖励函数
    reward_fn = Doc2JsonReward(text_weight=0.7, layout_weight=0.3)

    # 计算奖励
    reward, _, _ = reward_fn(
        json.dumps(pred_data, ensure_ascii=False),
        json.dumps(gt_data, ensure_ascii=False),
    )

    print(
        f"文本相似度奖励: {reward_fn.calculate_text_similarity_reward(pred_data, gt_data):.4f}"
    )
    print(f"布局奖励: {reward_fn.calculate_layout_reward(pred_data, gt_data):.4f}")
    print(f"总奖励: {reward[0]:.4f}")

    # 测试单个类别的IoU计算
    table_iou = reward_fn.calculate_category_iou(pred_data, gt_data, "table")
    print(f"Table类别的IoU: {table_iou:.4f}")

    # 测试仅bbox/category格式（无"text"字段）
    pred_data_no_text = [
        {"bbox": [12, 12, 98, 32], "category": "title"},
        {
            "bbox": [15, 42, 195, 105],
            "category": "paragraph",
        },
        {"bbox": [8, 108, 305, 205], "category": "table"},
        {"bbox": [45, 208, 255, 305], "category": "table"},
    ]
    gt_data_no_text = [
        {"bbox": [10, 10, 100, 30], "category": "title"},
        {
            "bbox": [10, 40, 200, 100],
            "category": "paragraph",
        },
        {"bbox": [10, 110, 300, 200], "category": "table"},
        {"bbox": [50, 210, 250, 300], "category": "table"},
    ]

    reward2, _, _ = reward_fn(
        json.dumps(pred_data_no_text, ensure_ascii=False),
        json.dumps(gt_data_no_text, ensure_ascii=False),
    )
    print(
        f"仅bbox/category格式 — 总奖励: {reward2[0]:.4f}, 文本: {reward2[1]:.4f}, 布局: {reward2[2]:.4f}"
    )
