# instruction_dataset_hf_verl.py
import argparse, os, json
from typing import Dict, Any, List, Optional
from copy import deepcopy
from datasets import load_dataset, Dataset
from tqdm import tqdm

INSTRUCTION_DEFAULT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer within \\boxed{}. Respond in the following format:\n"
    "<think>\n...\n</think>\n<answer>\n...\n</answer>"
)

# Columns we keep verbatim at top-level if present
TOP_KEEP = {"prompt", "data_source", "ability", "reward_model"}

def _text_from_content(c):
    """Normalize content to str if it's a list of segments."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for seg in c:
            if isinstance(seg, dict):
                if isinstance(seg.get("text"), str):
                    parts.append(seg["text"])
                elif isinstance(seg.get("content"), str):
                    parts.append(seg["content"])
        if parts:
            return "\n".join(parts)
    return c  # return as-is (rare)

def prepend_instruction_to_first_user(prompt: List[dict], instruction: str) -> List[dict]:
    """Return a new prompt list with instruction prepended to the FIRST user message."""
    newp = deepcopy(prompt)
    # find first user message; if none, first message
    idx = None
    for i, m in enumerate(newp):
        if isinstance(m, dict) and m.get("role") == "user":
            idx = i
            break
    if idx is None and len(newp) > 0:
        idx = 0

    if idx is not None:
        msg = newp[idx]
        old = _text_from_content(msg.get("content", ""))
        if not isinstance(old, str):
            old = str(old)
        msg["content"] = f"{instruction}\n\n{old}".strip()
    else:
        # no messages? create a single user message
        newp = [{"role": "user", "content": instruction}]
    return newp

def make_sample(row: Dict[str, Any], idx: int, split: str, instruction: str, default_ability="math") -> Optional[Dict[str, Any]]:
    # Require prompt list
    prompt = row.get("prompt")
    if not (isinstance(prompt, list) and prompt):
        # If prompt is a JSON string or plain string, wrap minimally
        if isinstance(prompt, str) and prompt.strip():
            prompt = [{"role": "user", "content": f"{instruction}\n\n{prompt.strip()}"}]
        else:
            return None

    new_prompt = prepend_instruction_to_first_user(prompt, instruction)

    # Pass through reward_model exactly (keep ground_truth)
    rm = row.get("reward_model", {})
    if not isinstance(rm, dict):
        rm = {}
    # Ensure keys exist and are strings (VERL-friendly)
    rm.setdefault("style", "rule")
    if "ground_truth" not in rm or rm["ground_truth"] is None:
        rm["ground_truth"] = ""  # only if truly missing; originals already have it

    # Pass-through data_source / ability with sensible defaults
    data_source = row.get("data_source", split)
    ability = row.get("ability", default_ability)

    # Everything else -> extra_info (so we don't drop useful fields like 'level', 'type', ...)
    extra_info = {"split": split, "index": idx}
    for k, v in row.items():
        if k not in TOP_KEEP:
            extra_info[k] = v

    return {
        "prompt": new_prompt,
        "data_source": data_source,
        "ability": ability,
        "reward_model": rm,
        "extra_info": extra_info,
    }

def convert_file(in_path: str, split: str, instruction: str) -> Dataset:
    ds = load_dataset("parquet", data_files=in_path)["train"]
    out = []
    dropped = 0
    for i in tqdm(range(len(ds)), desc=f"Embedding {split}"):
        smp = make_sample(ds[i], i, split, instruction)
        if smp is None:
            dropped += 1
            continue
        out.append(smp)

    if not out:
        raise RuntimeError(f"No rows produced from {in_path}; check schema.")
    print(f"[INFO] {split}: kept={len(out)}, dropped(no_prompt)={dropped}")
    return Dataset.from_list(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True)
    ap.add_argument("--val_in",   required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--instruction", default=INSTRUCTION_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = convert_file(args.train_in, "train", args.instruction)
    val_ds   = convert_file(args.val_in,   "val",   args.instruction)

    train_p = os.path.join(args.out_dir, "train.parquet")
    val_p   = os.path.join(args.out_dir, "val.parquet")

    train_ds.to_parquet(train_p)
    val_ds.to_parquet(val_p)

    print(f"[DONE] Saved {train_p} (rows={len(train_ds)})")
    print(f"[DONE] Saved {val_p}   (rows={len(val_ds)})")

if __name__ == "__main__":
    main()
