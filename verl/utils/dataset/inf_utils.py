import json
import re
from typing import Dict, List, Tuple
from PIL import Image
from qwen_vl_utils import smart_resize


def load_image(image: str) -> Image.Image:
    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_images(images: str | List[str]) -> List[Image.Image]:
    if isinstance(images, str):
        images = [images]
    images = [load_image(image) for image in images]
    return images


def get_norm_shapes(
    images: List[Image.Image], patch_factor: int, min_pixels: int, max_pixels: int
) -> List:
    norm_shapes = []
    for image in images:
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        norm_shapes.append([resized_width, resized_height])
    return norm_shapes


def normalize_bbox(
    objects: Dict,
    images: List[Image.Image],
    norm_bbox: str,
    patch_factor: int,
    min_pixels: int,
    max_pixels: int,
) -> None:
    objects["width"] = [image.width for image in images]
    objects["height"] = [image.height for image in images]
    norm_shapes = get_norm_shapes(images, patch_factor, min_pixels, max_pixels)

    bbox_list = objects["bbox"]
    width_list = objects["width"]
    height_list = objects["height"]
    bbox_type = objects.pop("bbox_type", None) or "real"
    image_id_list = objects.pop("image_id", None) or []
    image_id_list += [0] * (len(bbox_list) - len(image_id_list))
    for bbox, image_id in zip(bbox_list, image_id_list):
        if bbox_type == "norm1":
            width, height = 1, 1
        else:
            width, height = width_list[image_id], height_list[image_id]
        for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):
            if norm_bbox == "norm1000":
                norm_width, norm_height = 1000, 1000
            elif norm_bbox == "none":
                norm_width, norm_height = norm_shapes[image_id]
            bbox[2 * i] = int(round(x / width * norm_width))
            bbox[2 * i + 1] = int(round(y / height * norm_height))


def _split_str_by_regex(text: str, regex_delimiters: List[str]) -> List[str]:
    combined_pattern = "|".join(f"({pattern})" for pattern in regex_delimiters)
    parts = re.split(combined_pattern, text, flags=re.DOTALL)
    parts = [part for part in parts if part is not None]
    if parts[0] == "":
        parts.pop(0)
    else:
        parts.insert(0, "")
    assert len(parts) % 2 == 0, f"result: {parts}"
    assert "".join(parts) == text, f"split_result: {parts}, text: {text}"
    return parts


def split_str_parts_by(
    text: str, delimiters: List[str], regex_mode: bool = False
) -> List[Dict[str, str]]:
    """Split the text field into parts.

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text in list of dicts.
    """
    assert isinstance(text, str), f"text: {text}"
    delimiters_origin = delimiters
    if not regex_mode:
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    parts = _split_str_by_regex(text, delimiters) if delimiters else ["", text]
    res = []
    if regex_mode:
        parts = [part for part in parts if part]
        for part in parts:
            for delimiter, delimiter_origin in zip(delimiters, delimiters_origin):
                if re.match(delimiter, part, re.DOTALL):
                    break
            else:
                delimiter_origin = ""
            res.append({"key": delimiter_origin, "content": part})
    else:
        for key, content in zip(parts[::2], parts[1::2]):
            res.append({"key": key, "content": content})
    return res


def _split_special_tokens(
    context: str,
) -> List[str]:
    """Split special tokens, for example `<ref-object>`, `<bbox>`, this will help the replace_tag operation"""
    special_tokens = ["<ref-object>", "<bbox>"]

    res: List[str] = []
    contexts = []
    assert isinstance(context, str)
    for d in split_str_parts_by(context, special_tokens):
        contexts.extend([d["key"], d["content"]])
    contexts = [c for c in contexts if c]
    res.extend(contexts)

    return res


def _get_bbox_str(bbox: List[int]) -> str:
    point = []
    for x, y in zip(bbox[::2], bbox[1::2]):
        point.append(f"({x},{y})")
    return ",".join(point)


def replace_ref(ref: str, bbox_format: str = "new") -> List[str]:
    if bbox_format == "legacy":
        return [f"<|object_ref_start|>{ref}<|object_ref_end|>"]
    else:
        return [ref]


def replace_bbox(bbox: List[int], bbox_format: str = "new") -> List[str]:
    if bbox_format == "legacy":
        return [f"<|box_start|>{_get_bbox_str(bbox)}<|box_end|>"]
    else:
        return [str(bbox)]


def _merge_special_tokens(
    context_list: List[str],
    objects: Dict,
    bbox_format: str = "new",
) -> str:
    """Merge context list by replacing special tokens."""

    res = []
    ref = objects.get("ref") or []
    bbox = objects.get("bbox") or []
    ref_idx = bbox_idx = 0
    for context in context_list:
        if context == "<ref-object>" and ref_idx < len(ref):
            idx = ref_idx
            c_list = replace_ref(ref[idx], bbox_format)
            ref_idx += 1
        elif context == "<bbox>" and bbox_idx < len(bbox):
            idx = bbox_idx
            c_list = replace_bbox(bbox[idx], bbox_format)
            bbox_idx += 1
        else:
            c_list = [context]
        res += c_list

    return "".join(res)


def replace_special_tokens(context, objects, bbox_format="new"):
    """Update context with special tokens."""
    context_list = _split_special_tokens(context)
    res = _merge_special_tokens(context_list, objects, bbox_format=bbox_format)
    return res


if __name__ == "__main__":
    context = '```json\n[{"bbox": <bbox>, "category": "<ref-object>", "text": "[香·格·里·拉]\\nShangri-La..."}, {"bbox": <bbox>, "category": "<ref-object>", "text": ""}]\n```'
    objects = {
        "ref": ["header", "figure"],
        "bbox": [[282, 6, 404, 49], [40, 62, 328, 252]],
    }
    print(context)
    context_list = _split_special_tokens(context)
    print(json.dumps(context_list, indent=2, ensure_ascii=False))

    print(objects)
    res = _merge_special_tokens(context_list, objects, bbox_format="new")
    print(res)
