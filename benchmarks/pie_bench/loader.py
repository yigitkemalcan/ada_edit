"""
PIE-Bench loader.

PIE-Bench ships separately from the `cure-lab/PnPInversion` repo — it
is gated behind a Google Form linked in that repo's README. After
download, extract the archive so it looks like:

    data/
      mapping_file.json           # one entry per sample, keyed by 12-digit id
      annotation_images/
        0_random_140/...
        1_change_object_80/...
        ...

Our loader accepts either that exact layout or one wrapped in an
extra directory (e.g. `data/PIE-Bench/data/mapping_file.json` if you
cloned the PnPInversion repo first and then unpacked the dataset
inside it).

Each mapping_file.json entry has:

    image_path          : relative path under annotation_images/
    original_prompt     : source prompt, with [brackets] marking edit target
    editing_prompt      : target prompt, same bracket convention
    editing_instruction : natural language instruction
    editing_type_id     : "0".."9" (stored as string in the JSON)
    blended_word        : "src_word tgt_word" (space-separated pair)
    mask                : flat RLE list [start, length, start, length, ...]
                          decoded to a 512x512 binary array

We keep the GT mask (PIE-Bench RLE) because the AdaEdit paper reports
background-preservation metrics against exactly this mask, and we want
our numbers to be comparable.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


# Collapse PIE-Bench's 10 editing_type_id values onto AdaEdit's 4 buckets.
# category 8 (background) is excluded by default at the sampler level;
# the mapping is still provided in case the user wants it.
PIE_BENCH_TO_ADAEDIT: Dict[int, str] = {
    0: "change",  # random mix
    1: "change",  # change object
    2: "add",     # add object
    3: "remove",  # delete object
    4: "change",  # content
    5: "change",  # pose
    6: "change",  # color
    7: "change",  # material
    8: "change",  # background (skipped by default)
    9: "style",   # style
}


@dataclass
class PIESample:
    key: str                     # JSON key, e.g. "000000000000"
    editing_type_id: int
    image_path: str              # absolute path on disk
    source_prompt: str           # brackets stripped
    target_prompt: str           # brackets stripped
    edit_object: str             # source word from blended_word
    edit_type: str               # AdaEdit bucket (change/add/remove/style)
    mask: np.ndarray             # HxW float32, 1=edit region, 0=background
    blended_word: str            # raw "src tgt" pair from JSON


def _default_root() -> str:
    """Project convention: `<repo>/data/PIE-Bench/` for the unpacked dataset."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo_root, "data", "PIE-Bench")


def _resolve_paths(root: Optional[str]) -> Dict[str, str]:
    """
    Find mapping_file.json and the annotation_images directory. Accepts:

      - a path that directly contains `mapping_file.json`
      - a path containing a `data/` subdir with the dataset inside
    """
    if root is None:
        root = _default_root()
    candidates = [root, os.path.join(root, "data")]
    for base in candidates:
        mapping = os.path.join(base, "mapping_file.json")
        images = os.path.join(base, "annotation_images")
        if os.path.isfile(mapping) and os.path.isdir(images):
            return {"mapping": mapping, "images": images, "base": base}
    raise FileNotFoundError(
        f"Could not find mapping_file.json + annotation_images/ under {root!r}. "
        "PIE-Bench is gated behind the Google Form linked in the "
        "cure-lab/PnPInversion README. Download it and extract so that "
        "mapping_file.json + annotation_images/ live directly under the path "
        "you pass as data_root= (default: <repo>/data/PIE-Bench/)."
    )


def _mask_decode(encoded: List[int], shape=(512, 512)) -> np.ndarray:
    """
    Reproduces PnPInversion's mask_decode. RLE format: pairs of
    (flat_start_index, run_length). Returns a float32 HxW array in {0,1}
    with border pixels forced to 1 (matching PnPInversion's
    annotation-error workaround so our numbers stay comparable).
    """
    h, w = shape
    length = h * w
    arr = np.zeros(length, dtype=np.float32)
    for i in range(0, len(encoded), 2):
        start = encoded[i]
        run = min(encoded[i + 1], length - start)
        arr[start:start + run] = 1.0
    arr = arr.reshape(h, w)
    arr[0, :] = 1.0
    arr[-1, :] = 1.0
    arr[:, 0] = 1.0
    arr[:, -1] = 1.0
    return arr


def _strip_brackets(p: str) -> str:
    return p.replace("[", "").replace("]", "")


def _last_word(text: str) -> str:
    """Last non-empty alphanumeric word of a string (brackets stripped)."""
    stripped = _strip_brackets(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", stripped)
    return tokens[-1] if tokens else ""


def _derive_edit_object(
    blended_word: str,
    original_prompt: str,
    editing_prompt: str,
) -> str:
    """
    Return an edit-object word that can be found in the (unbracketed)
    source prompt. AdaEdit's mask extraction tokenizes this against the
    source prompt — if it is empty or absent the whole Phase-1 inversion
    crashes on an IndexError in layers.py:289.

    Priority:
      1. First whitespace-separated token of ``blended_word``
         (what PIE-Bench populates for ~78% of entries).
      2. Last word of the bracketed span in ``original_prompt``
         (change-object / remove-object edits annotate the target word
         this way even when blended_word is blank).
      3. Last content word of the unbracketed source prompt
         (add / style edits have no source-side bracket, so we pick the
         most salient noun — the final token of the source prompt —
         to give the cross-attn mask *something* to anchor on).
    """
    bw = blended_word.strip() if blended_word else ""
    if bw:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", bw)
        if tokens:
            return tokens[0]

    src_brackets = _BRACKET_RE.findall(original_prompt or "")
    if src_brackets:
        w = _last_word(src_brackets[-1])
        if w:
            return w

    return _last_word(original_prompt or editing_prompt or "")


def load_sample(
    key: str,
    item: Dict,
    paths: Dict[str, str],
) -> PIESample:
    type_id = int(item["editing_type_id"])
    image_path = os.path.join(paths["images"], item["image_path"])
    mask = _mask_decode(item["mask"])
    return PIESample(
        key=key,
        editing_type_id=type_id,
        image_path=image_path,
        source_prompt=_strip_brackets(item["original_prompt"]),
        target_prompt=_strip_brackets(item["editing_prompt"]),
        edit_object=_derive_edit_object(
            item.get("blended_word", ""),
            item.get("original_prompt", ""),
            item.get("editing_prompt", ""),
        ),
        edit_type=PIE_BENCH_TO_ADAEDIT.get(type_id, "change"),
        mask=mask,
        blended_word=item.get("blended_word", ""),
    )


def iter_samples(
    root: Optional[str] = None,
    include_categories: Optional[List[int]] = None,
    exclude_categories: Optional[List[int]] = (8,),
) -> Iterator[PIESample]:
    """Stream every PIE-Bench sample subject to category filters."""
    paths = _resolve_paths(root)
    with open(paths["mapping"], "r") as f:
        mapping = json.load(f)

    exclude = set(exclude_categories or [])
    include = set(include_categories) if include_categories is not None else None

    for key, item in mapping.items():
        type_id = int(item["editing_type_id"])
        if type_id in exclude:
            continue
        if include is not None and type_id not in include:
            continue
        yield load_sample(key, item, paths)


def sample_pie(
    n: int = 5,
    seed: int = 0,
    root: Optional[str] = None,
    include_categories: Optional[List[int]] = None,
    exclude_categories: Optional[List[int]] = (8,),
) -> List[PIESample]:
    """Draw `n` samples deterministically (seeded) after filtering."""
    pool = list(iter_samples(
        root=root,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
    ))
    if not pool:
        raise RuntimeError(
            "No PIE-Bench samples matched the requested filters."
        )
    rng = random.Random(seed)
    if n >= len(pool):
        return pool
    return rng.sample(pool, n)
