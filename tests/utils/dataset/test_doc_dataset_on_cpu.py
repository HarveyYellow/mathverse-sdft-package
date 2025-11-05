import unittest
# from unittest.mock import patch, MagicMock
import os
import sys
import json
import random
random.seed(42)  # You can use any integer value as the seed

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{cur_path}/..")

from omegaconf import DictConfig

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.doc_dataset import DocDataset


class TestDocDataset(unittest.TestCase):

    model_path = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-30B-A3B-Instruct"
    tokenizer = hf_tokenizer(model_path)
    processor = hf_processor(model_path)
    cfg = DictConfig({
        "prompt_key": "conversations",
        "image_patch_size": 16,
        "filter_overlong_prompts_workers": 1,
    })

    def test_load_single_data_files(self):
        """Test loading single data files."""

        data_files = "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/train_markdown_251103_sample3_v1.json"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertEqual(len(doc_dataset), 3)

    def test_load_multiple_data_files(self):
        """Test loading multiple data files."""

        data_files = [
            "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/train_markdown_251103_sample3_v1.json",
            "/home/ma-user/work/data_mllm/datasets/Infinity-Doc2/document_parsing/labels/train_markdown_251103_sample3_v1.json"
        ]
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
        )
        self.assertEqual(len(doc_dataset), 6)


if __name__ == '__main__':
    unittest.main()
