from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lpr_chars import BLANK_IDX, CHARS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RKNN LPRNet on a manifest-driven image folder.")
    parser.add_argument("--model", required=True, help="Path to .rknn model")
    parser.add_argument("--data-root", required=True, help="Folder containing manifest.tsv and images/")
    parser.add_argument("--manifest", default="manifest.tsv", help="Manifest path or filename under data-root")
    parser.add_argument("--img-width", type=int, default=94)
    parser.add_argument("--img-height", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick testing")
    parser.add_argument("--core-mask", default="auto", choices=["auto", "0", "1", "2"])
    return parser.parse_args()


def preprocess(image_path: Path, img_width: int, img_height: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    return np.expand_dims(image, axis=0)


def greedy_decode(logits: np.ndarray) -> str:
    arr = np.asarray(logits)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected logits shape: {arr.shape}")
    seq = np.argmax(arr[0], axis=0).tolist()
    cleaned: list[int] = []
    prev = seq[0]
    if prev != BLANK_IDX:
        cleaned.append(prev)
    for ch in seq[1:]:
        if ch == prev or ch == BLANK_IDX:
            if ch == BLANK_IDX:
                prev = ch
            continue
        cleaned.append(ch)
        prev = ch
    return "".join(CHARS[idx] for idx in cleaned)


def load_runtime(model_path: Path, core_mask: str) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    if core_mask == "auto":
        ret = rknn.init_runtime()
    else:
        mask = {
            "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,
            "2": RKNNLite.NPU_CORE_2,
        }[core_mask]
        ret = rknn.init_runtime(core_mask=mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    return rknn


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = data_root / manifest
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows to evaluate")

    rknn = load_runtime(Path(args.model).expanduser().resolve(), args.core_mask)

    total = 0
    correct = 0
    by_type_total = Counter()
    by_type_correct = Counter()
    errors: list[tuple[str, str, str, str]] = []

    for idx, row in enumerate(rows, start=1):
        image_path = data_root / row["image"]
        pred = greedy_decode(rknn.inference(inputs=[preprocess(image_path, args.img_width, args.img_height)])[0])
        gt = row["label"]
        plate_type = row.get("plate_type", "unknown")
        total += 1
        by_type_total[plate_type] += 1
        if pred == gt:
            correct += 1
            by_type_correct[plate_type] += 1
        elif len(errors) < 20:
            errors.append((row["image"], plate_type, gt, pred))

        if idx % 20 == 0 or idx == len(rows):
            print(f"[INFO] processed={idx}/{len(rows)} exact_acc={correct / max(total, 1):.4f}")

    print(f"[INFO] total={total}")
    print(f"[INFO] exact_match_acc={correct / max(total, 1):.4f}")
    print("[INFO] by_plate_type")
    for plate_type, count in sorted(by_type_total.items()):
        acc = by_type_correct[plate_type] / max(count, 1)
        print(f"  {plate_type}: count={count} acc={acc:.4f}")
    if errors:
        print("[INFO] sample_errors")
        for image, plate_type, gt, pred in errors:
            print(f"  {plate_type}\t{image}\tgt={gt}\tpred={pred}")

    rknn.release()


if __name__ == "__main__":
    main()
