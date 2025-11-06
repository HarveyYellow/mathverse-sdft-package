import unittest
import sys
import os
import numpy as np

# 添加当前目录到路径，以便导入奖励函数模块

cur_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{cur_dir}/../../..")

from examples.inf.reward_functions.infinity_parser2_seq_grpo_reward_base import DocumentParserReward, truncate_last_incomplete_element, extract_json_content 
from examples.inf.reward_functions.infinity_parser2_seq_grpo_reward_func import compute_score

class TestDocumentParserReward(unittest.TestCase):
    
    def setUp(self):
        """设置测试用例的公共部分"""
        self.reward_fn = DocumentParserReward(text_weight=0.5, layout_weight=0.5)
    
    def test_perfect_match(self):
        """测试完全匹配的情况"""
        gt = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"},
            {"bbox": [10, 110, 300, 200], "category": "table", "text": "表格数据"}
        ]
        
        pred = gt.copy()  # 完全相同的预测
        
        text_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred)
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        total_reward, _, _ = self.reward_fn(gt, pred)
        
        self.assertAlmostEqual(text_reward, 1.0, places=4)
        self.assertAlmostEqual(layout_reward, 1.0, places=4)
        self.assertAlmostEqual(total_reward, 1.0, places=4)
    
    def test_granularity_case(self):
        """测试粒度情况：gt中一个大的bbox，pred中多个小bbox"""
        # gt: 一个大的表格区域
        gt = [
            {"bbox": [0, 0, 100, 100], "category": "table", "text": "表格内容"}
        ]
        
        # pred: 将大表格拆分为4个小表格
        pred = [
            {"bbox": [0, 0, 50, 50], "category": "table", "text": "表格内容1"},
            {"bbox": [50, 0, 100, 50], "category": "table", "text": "表格内容2"},
            {"bbox": [0, 50, 50, 100], "category": "table", "text": "表格内容3"},
            {"bbox": [50, 50, 100, 100], "category": "table", "text": "表格内容4"}
        ]
        
        # 计算布局奖励
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # 在这种情况下，布局应该几乎完美匹配
        self.assertAlmostEqual(layout_reward, 1.0, places=4)
        
        # 文本奖励应该较低，因为文本被拆分
        text_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred)
        self.assertLess(text_reward, 0.5)
        
    def test_compute_score(self):
        """测试 json extract 和 truncation func：gt中一个大的bbox，pred中多个小bbox"""
        # gt: 一个大的表格区域
        gt = '```json\n[{"bbox": [0, 0, 100, 100], "category": "table", "text": "表格内容"}]\n```'
        
        # pred: 2个子表格和大表格重合
        pred = '```json\n[{"bbox": [0, 0, 50, 50], "category": "table", "text": "表格"},{"bbox": [50, 50, 150, 150], "category": "table", "text": "内容"},{"bbox": [50, 50, 150, 150],'
        
        reward_dict = compute_score(pred, gt)
        # 计算布局奖励
        layout_reward = reward_dict["layout_reward"]
        
        # 计算预期的IoU
        # 交集: [0, 0, 50, 50] [50, 50, 100, 100]   面积 = 2500 + 2500 = 5000
        # 并集: [0, 0, 100, 100] [100, 100, 150, 150] [50, 100, 150, 150] [100, 50, 150, 150] 面积 = 10000 + 3*2500 = 17500
        # IoU = 5000/17500 ≈ 2/7
        expected_iou = 5000 / 17500
        
        # 在这种情况下，布局应该几乎完美匹配
        self.assertAlmostEqual(layout_reward, 2/7, places=4)
        
        # 文本奖励应该较低，因为文本被拆分
        text_reward = reward_dict["text_reward"]
        self.assertAlmostEqual(text_reward, 2/3, places=4)
    
    def test_granularity_case2(self):
        """测试粒度情况：gt中一个大的bbox，pred中多个小bbox"""
        # gt: 一个大的表格区域
        gt = [
            {"bbox": [0, 0, 100, 100], "category": "table", "text": "表格内容"}
        ]
        
        # pred: 2个子表格和大表格重合
        pred = [
            {"bbox": [0, 0, 50, 50], "category": "table", "text": "表格"},
            {"bbox": [50, 50, 150, 150], "category": "table", "text": "内容"}
        ]
        
        # 计算布局奖励
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # 计算预期的IoU
        # 交集: [0, 0, 50, 50] [50, 50, 100, 100]   面积 = 2500 + 2500 = 5000
        # 并集: [0, 0, 100, 100] [100, 100, 150, 150] [50, 100, 150, 150] [100, 50, 150, 150] 面积 = 10000 + 3*2500 = 17500
        # IoU = 5000/17500 ≈ 2/7
        expected_iou = 5000 / 17500
        
        # 在这种情况下，布局应该几乎完美匹配
        self.assertAlmostEqual(layout_reward, 2/7, places=4)
        
        # 文本奖励应该较低，因为文本被拆分
        text_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred)
        self.assertAlmostEqual(text_reward, 2/3, places=4)
    
    def test_missing_category(self):
        """测试类别缺失的情况"""
        gt = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"},
            {"bbox": [10, 110, 300, 200], "category": "table", "text": "表格数据"}
        ]
        
        # pred缺少table类别
        pred = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"}
        ]
        
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # table类别在pred中缺失，所以table类别的IoU为0
        # 三个类别：title, paragraph, table
        # title和paragraph完美匹配，table为0
        expected_iou = (1.0 + 1.0 + 0.0) / 3
        self.assertAlmostEqual(layout_reward, expected_iou, places=4)
    
    def test_extra_category(self):
        """测试pred中有额外类别的情况"""
        gt = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"}
        ]
        
        # pred有额外的table类别
        pred = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"},
            {"bbox": [10, 110, 300, 200], "category": "table", "text": "表格数据"}
        ]
        
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # table类别在gt中缺失，所以table类别的IoU为0
        # 三个类别：title, paragraph, table
        # title和paragraph完美匹配，table为0
        expected_iou = (1.0 + 1.0 + 0.0) / 3
        self.assertAlmostEqual(layout_reward, expected_iou, places=4)
    
    def test_partial_overlap(self):
        """测试部分重叠的情况"""
        gt = [
            {"bbox": [0, 0, 100, 100], "category": "table", "text": "表格内容"}
        ]
        
        # pred的bbox与gt有50%重叠
        pred = [
            {"bbox": [50, 0, 150, 100], "category": "table", "text": "表格内容"}
        ]
        
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # 计算预期的IoU
        # 交集: [50, 0, 100, 100] -> 面积: 50*100 = 5000
        # 并集: [0, 0, 150, 100] -> 面积: 150*100 = 15000
        # IoU = 5000/15000 ≈ 0.3333
        expected_iou = 5000 / 15000
        self.assertAlmostEqual(layout_reward, expected_iou, places=4)
    
    def test_multiple_bboxes_same_category(self):
        """测试同一类别有多个bbox的情况"""
        gt = [
            {"bbox": [0, 0, 50, 50], "category": "table", "text": "表格1"},
            {"bbox": [50, 0, 100, 50], "category": "table", "text": "表格2"}
        ]
        
        # pred的bbox与gt有重叠但不同
        pred = [
            {"bbox": [25, 0, 75, 50], "category": "table", "text": "表格合并"}
        ]
        
        layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        
        # 计算预期的IoU
        # gt并集: [0, 0, 100, 50] -> 面积: 5000
        # pred并集: [25, 0, 75, 50] -> 面积: 2500
        # 交集: [25, 0, 75, 50] -> 面积: 2500
        # 并集: [0, 0, 100, 50] -> 面积: 5000
        # IoU = 2500/5000 = 0.5
        expected_iou = 0.5
        self.assertAlmostEqual(layout_reward, expected_iou, places=4)
    
    def test_text_similarity(self):
        """测试文本相似度计算"""
        gt = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"}
        ]
        
        # 完全相同的文本
        pred_same = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个段落内容"}
        ]
        
        # 完全不同的文本
        pred_different = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "不同的标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "不同的内容"}
        ]
        
        # 部分相同的文本
        pred_partial = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"},
            {"bbox": [10, 40, 200, 100], "category": "paragraph", "text": "这是一个不同的段落内容"}
        ]
        
        same_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred_same)
        different_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred_different)
        partial_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred_partial)
        
        self.assertAlmostEqual(same_reward, 1.0, places=4)
        self.assertLess(different_reward, 0.5)
        self.assertGreater(partial_reward, different_reward)
        self.assertLess(partial_reward, same_reward)
    
    def test_empty_inputs(self):
        """测试空输入的情况"""
        # 两个都是空
        gt = []
        pred = []
        
        # text_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred)
        # layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        total_reward, text_reward, layout_reward = self.reward_fn(gt, pred)
        
        self.assertAlmostEqual(text_reward, 1.0, places=4)
        self.assertAlmostEqual(layout_reward, 1.0, places=4)
        self.assertAlmostEqual(total_reward, 1.0, places=4)
        
        # gt有数据，pred为空
        gt = [
            {"bbox": [10, 10, 100, 30], "category": "title", "text": "文档标题"}
        ]
        pred = []
        
        # layout_reward = self.reward_fn.calculate_layout_reward(gt, pred)
        # text_reward = self.reward_fn.calculate_text_similarity_reward(gt, pred)
        total_reward, text_reward, layout_reward = self.reward_fn(gt, pred)
        # 所有类别的IoU都是0
        self.assertAlmostEqual(layout_reward, 0.0, places=4)
        self.assertAlmostEqual(text_reward, 0.0, places=4)
        self.assertAlmostEqual(total_reward, 0.0, places=4)


if __name__ == '__main__':
    
    # 运行测试
    unittest.main(verbosity=2)