import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from PIL import Image
import base64
import io
import re

try:
    from vllm import LLM, SamplingParams
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
    spec = importlib.util.spec_from_file_location(
        "infinity_parser2_multitask_rewards_v2", str(target)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将PIL图像转换为base64编码的字符串

    Args:
        image: PIL Image对象
        format: 输出格式，如 "PNG", "JPEG", "WEBP" 等

    Returns:
        base64编码的字符串，可以直接用于HTML img标签的src属性
    """
    # 创建一个字节流缓冲区
    buffer = io.BytesIO()

    # 将图像保存到缓冲区
    image.save(buffer, format=format)

    # 获取缓冲区中的字节数据
    img_bytes = buffer.getvalue()

    # 将字节数据转换为base64编码
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # 构建data URL格式
    # 常见的MIME类型映射
    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "GIF": "image/gif",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
    }

    mime_type = mime_types.get(format.upper(), "image/png")
    return f"data:{mime_type};base64,{img_base64}"


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
        self.model_path = "/home/ma-user/work/zuminghuang/03_projects/07_infinity_parser2/ms-swift/output/qwen3_vl_2b_sft_data_v1_8_mcore_pix4k_len32k/v0-20260107-193438/checkpoint-3828"
        try:
            # Keep tokenizer/processor as instance attributes for use below
            self.tokenizer = hf_tokenizer(self.model_path)
            self.processor = hf_processor(self.model_path)
        except Exception as e:
            self.skipTest(
                f"hf_tokenizer/hf_processor unavailable for {self.model_path}: {e}"
            )

    def test_vllm_full_chain_all_tasks(self):
        """
        For each multitask data_source:
         - create a minimal dataset entry used by DocDataset (conversations: [prompt, ground_truth])
         - build prompt via tokenizer.apply_chat_template
         - generate n=8 sequences via vllm
         - compute reward for each generated sequence via compute_score
        """
        data_files = "/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v1.8/train_v1.8_sample_5e-5.jsonl"
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
            prompt_text = self.tokenizer.decode(data_item["raw_prompt_ids"])
            match = re.search(r'<\|vision_end\|>(.*?)<\|im_end\|>', prompt_text, re.DOTALL)
            if match:
                prompt_text = match.group(1).strip()
            ground_truth = data_item["reward_model"]["ground_truth"]
            data_source = data_item["data_source"]

            params = {
                "max_tokens": 8192,
                "temperature": 0.6,
                "top_k": 50,
                "top_p": 0.95,
                "n": 8,
            }
            sampling_params = SamplingParams(**params)

            messages = [{"role": "user", "content": []}]
            for img in data_item["multi_modal_data"]["image"]:
                img = pil_to_base64(img)
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": img}}
                )
            # append contexts to images
            messages[0]["content"].append({"type": "text", "text": prompt_text})
            batched_messages = [messages]

            # generate n=8 outputs
            gen = client.chat(
                sampling_params=sampling_params, messages=batched_messages
            )
            gen_texts = []
            for gen_batch in gen:
                for out_item in getattr(gen_batch, "outputs", []):
                    txt = getattr(out_item, "text", None)
                    gen_texts.append(txt)

            gen_texts = gen_texts[:8]
            self.assertGreaterEqual(len(gen_texts), 1)

            # compute reward for each generated text
            scores = []
            for txt in gen_texts:
                res = self.mod.compute_score(txt, ground_truth, data_source=data_source)
                if isinstance(res, dict):
                    score = res.get("score", None)
                else:
                    score = res
                scores.append(score)

            try:
                # Assert scores are numeric and within [0,1]
                for s in scores:
                    self.assertIsNotNone(s)
                    self.assertIsInstance(s, float)
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s, 1.0)
            except:
                import pdb; pdb.set_trace();


if __name__ == "__main__":
    unittest.main()
