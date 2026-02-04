# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional, Any, List

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset import vision_utils
from verl.utils.dataset.inf_utils import (
    generate_schema,
    load_images,
    normalize_bbox,
    replace_special_tokens,
)

logger = logging.getLogger(__name__)

from importlib import metadata
from io import BytesIO
from PIL import Image
from qwen_vl_utils import fetch_image, vision_process


def custom_process_image(
    image: str | Image.Image, image_patch_size: int = 14
) -> Image.Image:
    ele = {"image": image}
    return fetch_image(ele, image_patch_size=image_patch_size)


vision_utils.process_image = custom_process_image


def set_pixels_for_vision_process(
    config: DictConfig,
) -> List[int]:
    from qwen_vl_utils import vision_process

    # set min pixels and max pixels
    if metadata.version("qwen-vl-utils") >= "0.0.14":
        patch_factor = 16 * 2
        min_pixels = config.get("min_pixels", 4 * patch_factor**2)
        max_pixels = config.get("max_pixels", 16384 * patch_factor**2)
        vision_process.IMAGE_MIN_TOKEN_NUM = min_pixels // (patch_factor**2)
        vision_process.IMAGE_MAX_TOKEN_NUM = max_pixels // (patch_factor**2)
    else:
        patch_factor = 14 * 2
        min_pixels = config.get("min_pixels", 4 * patch_factor**2)
        max_pixels = config.get("max_pixels", 16384 * patch_factor**2)
        vision_process.MIN_PIXELS = min_pixels
        vision_process.MAX_PIXELS = max_pixels

    return patch_factor, min_pixels, max_pixels


class DocDataset(RLHFDataset):
    """
    Load and preprocess RLHF data from JSON/JSONL/TXT files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to JSON/JSONL/TXT file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.patch_factor, self.min_pixels, self.max_pixels = set_pixels_for_vision_process(config)

        # set bbox format and norm bbox
        self.bbox_format = config.get("bbox_format", "new")
        self.norm_bbox = config.get("norm_bbox", "none")

        # allowed data_source list to keep (None means keep all)
        self.allowed_data_sources = config.get("allowed_data_sources", None)
        print(f"self.allowed_data_sources: {self.allowed_data_sources}")

        self.use_generated_schema = config.get("use_generated_schema", False)

        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def add_source_and_gt(self, dataset):

        def func(example):
            data_source = example.get("attributes", {}).get("subtask", "doc2json")
            ground_truth = example[self.prompt_key][-1]["value"]
            if example.get("objects") is not None:
                objects = example["objects"]
                # load images
                images = load_images(example[self.image_key])
                # normalize bbox
                normalize_bbox(
                    objects,
                    images,
                    self.norm_bbox,
                    self.patch_factor,
                    self.min_pixels,
                    self.max_pixels,
                )
                ground_truth = replace_special_tokens(
                    ground_truth, objects, self.bbox_format
                )
            example["data_source"] = data_source
            example["reward_model"] = {"style": "rule", "ground_truth": ground_truth}
            return example

        return dataset.map(func)

    def filter_data_sources(self, dataset):
        if self.allowed_data_sources is not None:
            dataset = dataset.filter(
                lambda x: x.get("attributes", {}).get("subtask", "doc2json") in self.allowed_data_sources
            )
            print(f"filtered data sources, dataset len: {len(dataset)}")
        return dataset

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            # Read JSON/JSONL/TXT files and cache.
            # Refer to https://git.infly.tech/inf_algo/ms-swift/-/blob/main/swift/llm/dataset/loader.py?ref_type=heads#L208-209
            ext = os.path.splitext(data_file)[1].lstrip(".")
            file_type = {"jsonl": "json", "txt": "text"}.get(ext) or ext
            if self.use_generated_schema:
                features = generate_schema(data_file)
                print(f"generated schema: {features}")
                dataframe = datasets.load_dataset(file_type, data_files=data_file, features=features)["train"]
            else:
                dataframe = datasets.load_dataset(file_type, data_files=data_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        # add data source and ground_truth
        self.dataframe = self.add_source_and_gt(self.dataframe)
        # filter long prompts
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        # filter data sources
        self.dataframe = self.filter_data_sources(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example.pop(self.prompt_key)[0]["value"]

        if self.image_key in example:
            if "<image>" not in prompt_str:
                prompt_str = f"<image>{prompt_str}"

            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    image_obj = Image.open(example[self.image_key][i - 1]).convert("RGB")
                    content_list.append({"type": "image", "image": image_obj})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            if "<video>" not in prompt_str:
                prompt_str = f"<video>{prompt_str}"

            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages. Derived from rl_dataset.py"""
        set_pixels_for_vision_process(config)

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos
