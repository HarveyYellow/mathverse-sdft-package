import unittest
import json
import os
import tempfile
import random
from datasets import Value, Sequence

from verl.utils.dataset.inf_utils import (
    get_type_feature,
    merge_schemas,
    generate_schema,
)


class TestSchemaGenerator(unittest.TestCase):

    def test_get_type_feature_basic(self):
        """测试基本类型的推断"""
        self.assertEqual(get_type_feature(123).dtype, "int64")
        self.assertEqual(get_type_feature(12.5).dtype, "float32")
        self.assertEqual(get_type_feature("hello").dtype, "string")
        self.assertEqual(get_type_feature(True).dtype, "bool")
        self.assertIsNone(get_type_feature(None))

    def test_get_type_feature_complex(self):
        """测试复杂嵌套结构的推断"""
        # 测试列表
        list_feat = get_type_feature([1, 2, 3])
        self.assertIsInstance(list_feat, Sequence)
        self.assertEqual(list_feat.feature.dtype, "int64")

        # 测试字典
        dict_val = {"a": 1, "b": "txt"}
        dict_feat = get_type_feature(dict_val)
        self.assertEqual(dict_feat["a"].dtype, "int64")
        self.assertEqual(dict_feat["b"].dtype, "string")

    def test_merge_schemas_null_handling(self):
        """核心测试：测试 Null 和 具体类型的合并 (模拟你的情况)"""
        schema_none = None
        schema_struct = {"x": Value("int64")}

        # 情况 A: 旧的是 None，新的是 Struct -> 应该变成 Struct
        result1 = merge_schemas(schema_none, schema_struct)
        self.assertEqual(result1, schema_struct)

        # 情况 B: 旧的是 Struct，新的是 None -> 应该保持 Struct
        result2 = merge_schemas(schema_struct, schema_none)
        self.assertEqual(result2, schema_struct)

    def test_merge_schemas_union_keys(self):
        """核心测试：测试列的并集 (Data 1 和 Data 2 字段不同)"""
        # Data 1 schema: {A, B}
        schema_1 = {"A": Value("int64"), "B": Value("string")}
        # Data 2 schema: {B, C}
        schema_2 = {"B": Value("string"), "C": Value("int64")}

        merged = merge_schemas(schema_1, schema_2)

        # 期望结果: {A, B, C}
        self.assertIn("A", merged)
        self.assertIn("B", merged)
        self.assertIn("C", merged)
        self.assertEqual(merged["A"].dtype, "int64")

    def test_merge_schemas_nested_recursive(self):
        """核心测试：递归合并嵌套字典"""
        # 假设第一行 D 是 {"x": 1} (没有 y)
        s1 = {"D": {"x": Value("int64")}}
        # 假设第二行 D 是 {"y": 2} (没有 x)
        s2 = {"D": {"y": Value("int64")}}

        merged = merge_schemas(s1, s2)

        # 期望 D 变成 {"x": int, "y": int}
        self.assertIsInstance(merged["D"], dict)
        self.assertIn("x", merged["D"])
        self.assertIn("y", merged["D"])


class TestIntegrationFile(unittest.TestCase):
    """集成测试：创建临时文件并运行完整流程"""

    def setUp(self):
        # 创建一个临时文件
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".jsonl", encoding="utf-8"
        )

    def tearDown(self):
        # 清理临时文件
        self.temp_file.close()
        os.remove(self.temp_file.name)

    def test_generate_schema_mixed_data(self):
        """
        场景模拟：
        Line 1: 数据1 (D为null)
        Line 2: 数据2 (D为复杂Struct)
        Line 3: 数据1 (D为null)
        期望：最终 Schema 能识别出 D 是 Struct，而不是 null 或 string
        """
        # 写入数据
        data = [
            json.dumps({"A": 1, "B": "row1", "D": None}),  # 数据 1
            json.dumps(
                {"A": 2, "B": "row2", "D": {"score": 99, "desc": "good"}}
            ),  # 数据 2 (定义了 Schema)
            json.dumps({"A": 3, "B": "row3", "D": None}),  # 数据 1
        ]
        self.temp_file.write("\n".join(data))
        self.temp_file.close()

        # 运行函数 (采样数大于行数，确保读到那行关键的 数据 2)
        features = generate_schema(self.temp_file.name, samples=10)

        # === 验证结果 ===
        print("\n[Test Result] Generated Features:", features)

        # 1. 验证 A, B 存在
        self.assertIn("A", features)
        self.assertIn("B", features)

        # 2. 验证 D 存在且类型正确
        self.assertIn("D", features)

        # 关键验证：D 应该是一个 Feature 字典 (即 Struct)，而不是 Value('string') 或 None
        # 注意：在 HuggingFace Features 中，Struct 表现为 Sequence 或 dict
        # 这里我们的 get_type_feature 返回的是 dict 结构
        d_feature = features["D"]
        self.assertIsInstance(d_feature, dict)

        # 3. 验证 D 内部的字段
        self.assertIn("score", d_feature)
        self.assertIn("desc", d_feature)
        self.assertEqual(d_feature["score"].dtype, "int64")
        self.assertEqual(d_feature["desc"].dtype, "string")


if __name__ == "__main__":
    unittest.main()
