import unittest
# from unittest.mock import patch, MagicMock
import os
import sys
import json
import random
random.seed(42)  # You can use any integer value as the seed
import re

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{cur_path}/..")

from PIL import Image, ImageDraw, ImageFont
from omegaconf import DictConfig

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.inf_dataset import DocDataset


def draw_bboxes_on_image(image_path, bboxes, output_path=None):
    """
    在图片上绘制bbox并保存结果。
    bboxes: [{"category": str, "bbox": [x1, y1, x2, y2]}, ...]
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for item in bboxes:
        category = item.get("category", "")
        x1, y1, x2, y2 = item["bbox"]

        # rescale to original image
        x1 = x1 / 1000 * image.width
        y1 = y1 / 1000 * image.height
        x2 = x2 / 1000 * image.width
        y2 = y2 / 1000 * image.height

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # 计算文字大小（兼容 Pillow ≥10）
        try:
            bbox_text = draw.textbbox((x1, y1), category, font=font)
            text_w, text_h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        except Exception:
            # 向后兼容旧版本 Pillow
            text_w, text_h = font.getsize(category)

        # 绘制文字背景与文字
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), category, fill="white", font=font)

    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_boxed{ext}"

    image.save(output_path, quality=95)
    print(f" 保存完成: {output_path}")
    return output_path


class TestDocDataset(unittest.TestCase):

    def setUp(self):
        self.model_path = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-30B-A3B-Instruct"
        self.tokenizer = hf_tokenizer(self.model_path)
        self.processor = hf_processor(self.model_path)
        self.cfg = DictConfig({
            "bbox_format": "new",
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": 1,
            "image_patch_size": 16,
            "max_pixels": 1344 * 1344,
            "max_prompt_length": 4096,
            "max_response_length": 4096,
            "norm_bbox": "norm1000",
            "prompt_key": "conversations",
            "shuffle": True,
        })

    def test_load_single_data_files(self):
        """Test loading single data files."""

        data_files = "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/val_markdown_251103_sample3_v1.json"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertEqual(len(doc_dataset), 3)
        self.assertTrue("raw_prompt" in doc_dataset[0])
        self.assertTrue("data_source" in doc_dataset[0])
        self.assertTrue("reward_model" in doc_dataset[0])
        self.assertEqual(doc_dataset[0]["reward_model"]["style"], "rule")
        self.assertTrue("ground_truth" in doc_dataset[0]["reward_model"])

        data_files = "/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v1.8/train_v1.8_sample_5pct.jsonl"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
            max_samples=1000,
        )
        self.assertEqual(len(doc_dataset), 1000)
        self.assertTrue("raw_prompt" in doc_dataset[0])
        self.assertTrue("data_source" in doc_dataset[0])
        self.assertTrue("reward_model" in doc_dataset[0])
        self.assertEqual(doc_dataset[0]["reward_model"]["style"], "rule")
        self.assertTrue("ground_truth" in doc_dataset[0]["reward_model"])
        data_sources = {item["data_source"] for item in doc_dataset}
        print(f"data_sources: {data_sources}")
        all_data_sources = {
            "doc2json",
            "doc2md",
            "text2md",
            "chart2code",
            "layout_analysis",
            "table2html",
            "table2md",
            "formula2latex",
            "chart2text",
            "chart2table",
            "chart2json",
            "chem2smiles",
            "docvqa"
        }
        self.assertTrue(data_sources.issubset(all_data_sources))

    def test_load_multiple_data_files(self):
        """Test loading multiple data files."""

        data_files = [
            "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/val_markdown_251103_sample3_v1.json",
            "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/val_markdown_251103_sample3_v1.json"
        ]
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertEqual(len(doc_dataset), 6)
        self.assertTrue("raw_prompt" in doc_dataset[0])
        self.assertTrue("data_source" in doc_dataset[0])
        self.assertTrue("reward_model" in doc_dataset[0])
        self.assertEqual(doc_dataset[0]["reward_model"]["style"], "rule")
        self.assertTrue("ground_truth" in doc_dataset[0]["reward_model"])

    def test_replace_special_tokens(self):
        """Test replace special tokens."""

        data_files = "/home/ma-user/work/datasets/Infinity-Doc2/document_parsing/labels/infinity_doc2_pdf2md_data_swift_sample3_v1.json"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertGreater(len(doc_dataset), 0)
        self.assertTrue("raw_prompt" in doc_dataset[0])
        self.assertTrue("data_source" in doc_dataset[0])
        self.assertTrue("reward_model" in doc_dataset[0])
        self.assertEqual(doc_dataset[0]["reward_model"]["style"], "rule")
        self.assertTrue("ground_truth" in doc_dataset[0]["reward_model"])
        self.assertTrue("<bbox>" not in doc_dataset[0]["reward_model"]["ground_truth"])
        self.assertTrue("<ref-object>" not in doc_dataset[0]["reward_model"]["ground_truth"])
        print(doc_dataset[0]["reward_model"]["ground_truth"])

    def test_norm_bbox(self):
        data_files = "/home/ma-user/work/datasets/Infinity-Doc2/document_parsing/labels/infinity_doc2_pdf2md_data_swift_sample3_v1.json"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertGreater(len(doc_dataset), 0)
        self.assertTrue("raw_prompt" in doc_dataset[0])
        self.assertTrue("data_source" in doc_dataset[0])
        self.assertTrue("reward_model" in doc_dataset[0])
        self.assertEqual(doc_dataset[0]["reward_model"]["style"], "rule")
        self.assertTrue("ground_truth" in doc_dataset[0]["reward_model"])
        self.assertTrue("<bbox>" not in doc_dataset[0]["reward_model"]["ground_truth"])
        self.assertTrue("<ref-object>" not in doc_dataset[0]["reward_model"]["ground_truth"])

        # draw bboxes on image
        image_path = json.load(open(data_files))[0]["images"][0]
        ground_truth = doc_dataset[0]["reward_model"]["ground_truth"]
        match = re.search(r'```json\n(.*)\n```', ground_truth, re.DOTALL)
        if match:
            match = match.group(1).strip()
        output_path = f"{cur_path}/vis_images/{os.path.basename(image_path)}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        draw_bboxes_on_image(image_path, json.loads(match), output_path=output_path)


if __name__ == '__main__':
    unittest.main()
