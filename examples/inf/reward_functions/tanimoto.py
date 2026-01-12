# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

# Disable RDKit logging to prevent console spam from invalid SMILES or dummy atoms
RDLogger.DisableLog("rdApp.*")


def extract_valid_content(text):

    patterns = [
        r"<smiles>(.*?)</smiles>",
    ]
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            text = matches.group(1)

    return text


def clean_cxsmiles(smiles_string):
    """
    Clean CXSMILES by removing the extended layer (everything after |).
    CXSMILES format: SMILES |extended_info|
    Example: CC(C)N |$R1;;;$| -> CC(C)N

    Also handles:
    - Whitespace after SMILES (before extended layer)
    - Multiple | characters
    """
    if not smiles_string:
        return smiles_string

    # Remove CXSMILES extended layer (everything after first |)
    if "|" in smiles_string:
        smiles_string = smiles_string.split("|")[0].strip()

    # Also handle space-separated format (SMILES followed by other info)
    # Be careful: some SMILES may have spaces in certain representations
    # Only split on space if what follows looks like extended info
    if " " in smiles_string:
        parts = smiles_string.split(" ", 1)
        # If first part looks like a valid SMILES (no obvious non-SMILES chars)
        # and second part starts with special chars, take first part only
        if len(parts) > 1 and parts[1].startswith(("$", "|", "c:", "f:")):
            smiles_string = parts[0]

    return smiles_string


def get_canonical_smiles(smiles):
    try:
        # 1. 解析：将 SMILES 字符串转化为分子对象（Graph）
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        # 2. 生成：将分子对象转回 SMILES，canonical=True 是默认开启的
        # RDKit 会自动计算原子的唯一排序
        canonical_s = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return canonical_s
    except Exception as e:
        return smiles


def parse_smiles(smiles_string):
    """
    Clean and parse a single SMILES string from model output.
    Handles:
    - Markdown code blocks (```plaintext ... ```)
    - CXSMILES extended layers (|...|)
    - Common prefixes (SMILES:, etc.)
    - Quotes and whitespace
    """
    if not smiles_string:
        return None

    # Remove markdown code block markers (```plaintext, ```, etc.)
    smiles_string = re.sub(r"```\w*\n?", "", smiles_string)
    smiles_string = re.sub(r"```", "", smiles_string)

    # Remove whitespace and control characters
    smiles_string = smiles_string.strip().replace("\n", "").replace("\r", "")

    # Extract SMILES if it's embedded in longer text (e.g., "SMILES: XXX")
    if ":" in smiles_string:
        parts = smiles_string.split(":")
        if len(parts) > 1:
            smiles_string = parts[-1].strip()

    # Remove quotes if present
    smiles_string = smiles_string.strip('"').strip("'")

    # Remove <smiles></smiles> tags
    match = re.search(r"<smiles>(.*?)</smiles>", smiles_string, re.DOTALL)
    if match:
        smiles_string = match.group(1).strip()

    # Clean CXSMILES extended layer
    smiles_string = clean_cxsmiles(smiles_string)

    # Get canonical smiles
    smiles_string = get_canonical_smiles(smiles_string)

    # Replace *
    if "*" in smiles_string:
        smiles_string = smiles_string.replace("*", "C")

    return smiles_string if smiles_string else None


def get_mol_from_smiles(smiles):
    """Get molecule file from smiles."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        # eval_logger.debug(f"Error parsing predicted SMILES '{smiles}': {e}")
        mol = None
    return mol


def tanimoto_similarity(pred, gt):
    """
    Computes Tanimoto similarity for a single molecule sample.
    """
    # Get predicted SMILES
    predicted_smiles = parse_smiles(pred)
    reference_smiles = parse_smiles(gt)

    # Get molecule file
    predicted_mol = get_mol_from_smiles(predicted_smiles)
    reference_mol = get_mol_from_smiles(reference_smiles)

    result = 0.0
    # If both molecules are valid, compute metrics
    if predicted_mol and reference_mol:
        try:
            predicted_fp = Chem.RDKFingerprint(predicted_mol)
            reference_fp = Chem.RDKFingerprint(reference_mol)
            result = DataStructs.FingerprintSimilarity(predicted_fp, reference_fp)
        except Exception as e:
            # logging.debug(f"Error computing Tanimoto for '{predicted_smiles}': {e}")
            pass

    return result


def tanimoto_reward(solution_str: str, ground_truth: str) -> float:
    try:
        # extract valid content
        solution_str = extract_valid_content(solution_str)
        ground_truth = extract_valid_content(ground_truth)

        reward = tanimoto_similarity(solution_str, ground_truth)
        return reward
    except Exception as e:
        return 0.0
