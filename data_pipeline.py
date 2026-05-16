import os
import json
import hashlib
from pathlib import Path
from typing import Iterable, List, Any

import tqdm
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

DATASET_NAME = "Roblox/luau_corpus"
DATA_ROOT = Path.cwd() / "data"
CLEAN_ROOT = DATA_ROOT / "clean"
TOKENIZER_PATH = DATA_ROOT / "tokenizer.json"

MIN_SCRIPT_LEN = 100
ROBLOX_KEYWORDS = [
    "game:GetService",
    "Instance.new",
    "require",
    "script.Parent",
    "Players",
    "workspace",
    "RunService",
]

VOCAB_SIZE = 32_000

_missing_warnings = 0


def _find_string_recursively(obj: Any, min_len: int = 20) -> str | None:
    if isinstance(obj, str):
        return obj.strip() if len(obj.strip()) >= min_len else None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_string_recursively(v, min_len)
            if found:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_string_recursively(v, min_len)
            if found:
                return found
    return None


def _extract_script(row: dict) -> str | None:
    global _missing_warnings
    possible_keys = ("content", "code", "text", "source", "script", "lua")
    for key in possible_keys:
        if key not in row:
            continue
        val = row[key]

        if isinstance(val, str) and val.strip():
            return val.strip()

        if isinstance(val, dict):
            for sub_key in ("text", "code", "content"):
                if sub_key in val and isinstance(val[sub_key], str) and val[sub_key].strip():
                    return val[sub_key].strip()

        if _missing_warnings < 5:
            print(f"[WARN] Row {row.get('id', '?')} has empty or non-string '{key}' field.")
            _missing_warnings += 1
        return None

    fallback = _find_string_recursively(row, min_len=20)
    if fallback:
        return fallback

    if _missing_warnings < 5:
        print(f"[WARN] Row {row.get('id', '?')} lacks any recognizable script field.")
        _missing_warnings += 1
    return None


def _train_tokenizer(scripts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    print("Training a fresh Byte Level BPE tokenizer …")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(scripts, trainer=trainer)
    
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Tokenizer saved to {TOKENIZER_PATH} (vocab size = {vocab_size})")
    return tokenizer


def _load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer not found at {path}.")
    return Tokenizer.from_file(str(path))


def _replace_unknown_chars(tokenizer: Tokenizer, txt: str) -> str:
    unk_id = tokenizer.token_to_id("<unk>")
    if unk_id is None:
        return txt
    cleaned = []
    for ch in txt:
        if tokenizer.token_to_id(ch) is None:
            cleaned.append("<unk>")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _validate_compatibility(tokenizer: Tokenizer, texts: Iterable[str]) -> None:
    unknown_chars: set[str] = set()
    for txt in texts:
        for ch in set(txt):
            if tokenizer.token_to_id(ch) is None:
                unknown_chars.add(ch)

    if unknown_chars:
        sample = ", ".join(list(unknown_chars)[:20])
        print(
            f"[WARN] Tokenizer cannot represent {len(unknown_chars)} unique character(s). "
            f"Examples: {sample} … They will be replaced with <unk> during tokenisation."
        )
    else:
        print(f"Tokenizer compatibility check passed (vocab size = {tokenizer.get_vocab_size()}).")


def _script_is_valid(script: str) -> bool:
    if len(script) < MIN_SCRIPT_LEN:
        return False
    for kw in ROBLOX_KEYWORDS:
        if kw in script:
            return True
    return False


def _write_script(dst_dir: Path, idx: int, script: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    filename = dst_dir / f"script_{idx:06d}.lua"
    new_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
    if filename.is_file():
        existing = filename.read_text(encoding="utf-8")
        if hashlib.sha256(existing.encode("utf-8")).hexdigest() == new_hash:
            return
    filename.write_text(script, encoding="utf-8")


def _process_split(
    split_name: str,
    ds_split,
    tokenizer: Tokenizer,
    out_root: Path,
) -> None:
    raw_iter = (_extract_script(row) for row in ds_split)

    sample_for_check: List[str] = []
    for script in raw_iter:
        if script is None:
            continue
        if len(sample_for_check) < 10_000:
            sample_for_check.append(script)
        else:
            break
    _validate_compatibility(tokenizer, sample_for_check)

    total = ds_split.num_rows
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with tqdm.tqdm(total=total, desc=f"Processing {split_name}", unit="script") as pbar:
        for idx, row in enumerate(ds_split):
            script = _extract_script(row)
            if script is None:
                pbar.update(1)
                continue

            script = _replace_unknown_chars(tokenizer, script)

            if _script_is_valid(script):
                _write_script(out_dir, idx, script)
            pbar.update(1)


def main() -> None:
    print("Loading Hugging Face dataset …")
    ds: DatasetDict = load_dataset(DATASET_NAME, split=None)

    print("Collecting scripts for tokenizer training …")
    training_scripts: List[str] = []
    max_training_scripts = 200_000
    source_split = ds.get("train") or ds.get("test")
    rows_inspected = 0
    for row in source_split:
        rows_inspected += 1
        script = _extract_script(row)
        if script is None:
            continue
        training_scripts.append(script)
        if len(training_scripts) >= max_training_scripts:
            break

    if not training_scripts:
        print("First pass found no usable scripts - trying a second pass.")
        training_scripts.clear()
        rows_inspected = 0
        for row in source_split:
            rows_inspected += 1
            script = _extract_script(row)
            if script is None:
                continue
            training_scripts.append(script)
            if len(training_scripts) >= max_training_scripts:
                break

    if not training_scripts:
        print("\nUnable to locate any Lua source code in the dataset.")
        for i, example in enumerate(source_split.select(range(min(5, source_split.num_rows)))):
            print(json.dumps(example, indent=2))
        raise RuntimeError(
            "Could not find any usable scripts to train the tokenizer. "
            "Check the field names in the dataset (see the printed examples above)."
        )

    tokenizer = _train_tokenizer(training_scripts, vocab_size=VOCAB_SIZE)
    tokenizer = _load_tokenizer(TOKENIZER_PATH)

    for split_name in ["train", "test"]:
        if split_name not in ds:
            print(f"Split '{split_name}' not present - skipping.")
            continue
        _process_split(split_name, ds[split_name], tokenizer, CLEAN_ROOT)

    print("\nFinished. Clean Luau files are in:", CLEAN_ROOT)


if __name__ == "__main__":
    main()
