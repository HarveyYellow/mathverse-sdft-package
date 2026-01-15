import json
import tempfile
import unittest
import importlib.util
from pathlib import Path


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

    def _write_tmp_json(self, entries):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
        json.dump(entries, tmp, ensure_ascii=False)
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_vllm_full_chain_all_tasks(self):
        """
        For each multitask data_source:
         - create a minimal dataset entry used by DocDataset (conversations: [prompt, ground_truth])
         - build prompt via tokenizer.apply_chat_template
         - generate n=8 sequences via vllm
         - compute reward for each generated sequence via compute_score
        """
        samples = {
            "doc2json": ("Extract key-value pairs from text", "[]"),
            "doc2md": ("Convert to markdown", "same text"),
            "text2md": ("Convert plain text to markdown", "same text"),
            "chart2code": ("Describe chart and output code", "print(1)"),
            "layout_analysis": ("Analyze layout", "[]"),
            "table2html": ("Convert table to HTML", "<table><tr><td>1</td></tr></table>"),
            "table2md": ("Convert table to Markdown", "| 1 |\n| - |\n| 1 |"),
            "formula2latex": ("Convert formula to LaTeX", "$x=1$"),
            "chart2text": ("Describe the chart in text", "hello world"),
            "chart2table": ("Convert chart to table", "| a |\n| - |\n| 1 |"),
            "chart2json": ("Convert chart to json", "{}"),
            "chem2smiles": ("Convert molecule to smiles", "C"),
            "docvqa": ("Answer question about document", "42"),
        }

        from verl.utils import hf_tokenizer, hf_processor
        from verl.utils.dataset.inf_dataset import DocDataset

        model_path = "/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-2B"
        try:
            tokenizer = hf_tokenizer(model_path)
            processor = hf_processor(model_path)
        except Exception as e:
            self.skipTest(f"hf_tokenizer/hf_processor unavailable for {model_path}: {e}")

        if not hasattr(tokenizer, "apply_chat_template"):
            self.skipTest("tokenizer does not implement apply_chat_template; cannot build prompts")

        try:
            from vllm import LLM
        except Exception as e:
            self.skipTest(f"vllm not available: {e}")

        client = LLM(model=model_path)

        cfg = {
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

        for ds, (prompt, ground_truth) in samples.items():
            with self.subTest(data_source=ds):
                entry = {
                    "conversations": [{"value": prompt}, {"value": ground_truth}],
                    "attributes": {"subtask": ds},
                }
                data_file = self._write_tmp_json([entry])

                doc_dataset = DocDataset(data_files=data_file, tokenizer=tokenizer, processor=processor, config=type("C",(object,),cfg))
                # take first item messages
                row = doc_dataset.dataframe[0]
                messages = row["conversations"]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                # generate n=8 outputs
                gen = client.generate(prompt=prompt_text, max_tokens=64, n=8)
                gen_texts = []
                for out in gen:
                    for item in out.outputs:
                        txt = getattr(item, "text", None)
                        if txt is None:
                            token_ids = getattr(item, "token_ids", None)
                            if token_ids is not None and hasattr(tokenizer, "decode"):
                                txt = tokenizer.decode(token_ids)
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


