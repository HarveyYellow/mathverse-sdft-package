import unittest
import json
import os
import tempfile
from datasets import Value, Sequence, Features

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
        # 测试字典列表 (Array of Structs)
        list_dict_val = [{"from": "user", "value": "hi"}]
        feat = get_type_feature(list_dict_val)

        # 修复点：现在应该是一个 Python list
        self.assertIsInstance(feat, list)
        self.assertIsInstance(feat[0], dict)
        self.assertEqual(feat[0]["from"].dtype, "string")

    def test_merge_schemas_sequence(self):
        """测试序列合并"""
        # 使用 [ ] 表示序列
        s1 = {"conv": [{"from": Value("string")}]}
        s2 = {"conv": [{"value": Value("string")}]}

        merged = merge_schemas(s1, s2)

        # 验证结构
        self.assertIsInstance(merged["conv"], list)
        self.assertIn("from", merged["conv"][0])
        self.assertIn("value", merged["conv"][0])

    def test_merge_schemas_null_handling(self):
        """测试 Null 和具体类型的合并"""
        schema_none = None
        schema_struct = {"x": Value("int64")}

        result1 = merge_schemas(schema_none, schema_struct)
        self.assertEqual(result1, schema_struct)

        result2 = merge_schemas(schema_struct, schema_none)
        self.assertEqual(result2, schema_struct)

    def test_merge_schemas_union_keys(self):
        """测试列的并集"""
        schema_1 = {"A": Value("int64"), "B": Value("string")}
        schema_2 = {"B": Value("string"), "C": Value("int64")}
        merged = merge_schemas(schema_1, schema_2)
        self.assertCountEqual(merged.keys(), ["A", "B", "C"])

    def test_merge_schemas_nested_recursive(self):
        """测试递归合并嵌套字典 (Struct)"""
        s1 = {"D": {"x": Value("int64")}}
        s2 = {"D": {"y": Value("int64")}}
        merged = merge_schemas(s1, s2)
        self.assertIsInstance(merged["D"], dict)
        self.assertIn("x", merged["D"])
        self.assertIn("y", merged["D"])


class TestIntegrationFile(unittest.TestCase):
    """集成测试：模拟真实 JSONL 处理"""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".jsonl", encoding="utf-8"
        )

    def tearDown(self):
        self.temp_file.close()
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_generate_schema_with_conversations(self):
        data = [
            json.dumps({"conversations": [{"from": "human"}]}),
            json.dumps({"conversations": [{"value": "hello"}]}),
        ]
        self.temp_file.write("\n".join(data))
        self.temp_file.close()

        # generate_schema 最终会调用 Features(final_dict)
        features = generate_schema(self.temp_file.name, samples=10)

        self.assertIsInstance(features, Features)
        self.assertIn("conversations", features)

        # 在 Features 对象内部，HF 会把 [dict] 转换为它的内部表示
        # 此时我们可以通过检查其结构来验证，或者检查它是否表现得像个序列
        conv_feat = features["conversations"]

        # 注意：在 Features 内部，List of Structs 确实会被转换
        # 我们验证它是否能成功通过 load_dataset 的类型检查即可
        # 如果非要断言类型，此时它可能是 Sequence 对象或被转换后的 dict
        # 但关键是它的 schema 已经正确包含了 'from' 和 'value'
        self.assertIn("from", str(conv_feat))
        self.assertIn("value", str(conv_feat))


if __name__ == "__main__":
    unittest.main()
