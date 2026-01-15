import json
import tempfile
import unittest
import importlib.util
from pathlib import Path

try:
    from vllm import LLM
except Exception as e:
    raise unittest.SkipTest(f"vllm not available: {e}")

from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.inf_dataset import DocDataset


def load_reward_module():
    repo_root = Path(__file__).resolve().parents[3]
    target = (
        repo_root
        / "examples"
        / "inf"
        / "reward_functions"
        / "infinity_parser2_multitask_rewards_v2.py"
    )
    spec = importlib.util.spec_from_file_location("infinity_parser2_multitask_rewards_v2", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMultitaskVLLMFullChain(unittest.TestCase):

    def setUp(self) -> None:
        self.mod = load_reward_module()
        self.cfg = {
            "bbox_format": "new",
            "filter_overlong_prompts_workers": 1,
            "image_patch_size": 16,
            "max_pixels": 1344 * 1344,
            "max_prompt_length": 4096,
            "max_response_length": 4096,
            "norm_bbox": "norm1000",
            "prompt_key": "conversations",
            "shuffle": False,
        }
        self.model_path = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-2B"
        try:
            # Keep tokenizer/processor as instance attributes for use below
            self.tokenizer = hf_tokenizer(model_path)
            self.processor = hf_processor(model_path)
        except Exception as e:
            self.skipTest(f"hf_tokenizer/hf_processor unavailable for {model_path}: {e}")

    def test_vllm_full_chain_all_tasks(self):
        """
        For each multitask data_source:
         - create a minimal dataset entry used by DocDataset (conversations: [prompt, ground_truth])
         - build prompt via tokenizer.apply_chat_template
         - generate n=8 sequences via vllm
         - compute reward for each generated sequence via compute_score
        """
        data_files = "/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v1.8/train_v1.8_sample_5pct.jsonl"
        doc_dataset = DocDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.cfg,
            max_samples=1000,
        )

        client = LLM(model=self.model_path)

        for data_item in doc_dataset:
            # Convert dataset item to prompt text, ground truth and data source.
            # DocDataset produces examples with a `conversations` field (list-like)
            if isinstance(data_item, dict):
                conversations = data_item.get("conversations") or data_item.get("prompt") or []
            else:
                conversations = []

            if not conversations or len(conversations) < 2:
                # skip malformed entries
                continue

            prompt_text = conversations[0]
            ground_truth = conversations[1]
            ds = data_item.get("data_source", None) if isinstance(data_item, dict) else None

            # generate n=8 outputs
            gen = client.generate(prompt=prompt_text, max_tokens=64, n=8)
            gen_texts = []
            for gen_batch in gen:
                for out_item in getattr(gen_batch, "outputs", []):
                    txt = getattr(out_item, "text", None)
                    if txt is None:
                        token_ids = getattr(out_item, "token_ids", None)
                        if token_ids is not None and hasattr(self.tokenizer, "decode"):
                            txt = self.tokenizer.decode(token_ids)
                        else:
                            txt = ""
                    gen_texts.append(txt)

            gen_texts = gen_texts[:8]
            self.assertGreaterEqual(len(gen_texts), 1)

            # compute reward for each generated text
            scores = []
            for txt in gen_texts:
                res = self.mod.compute_score(txt, ground_truth, data_source=ds)
                if isinstance(res, dict):
                    score = res.get("score", None)
                else:
                    score = res
                scores.append(score)

            # Assert scores are numeric and within [0,1]
            for s in scores:
                self.assertIsNotNone(s)
                self.assertIsInstance(s, float)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()


