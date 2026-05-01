from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


SCRIPT_DIR = Path(__file__).resolve().parent
LPR_DIR = SCRIPT_DIR.parent / "lpr_work" / "project"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if LPR_DIR.exists() and str(LPR_DIR) not in sys.path:
    sys.path.insert(0, str(LPR_DIR))

from lpr_chars import BLANK_IDX, CHARS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image LPRNet inference with RKNNLite on RK3568.")
    parser.add_argument("--model", required=True, help="Path to .rknn model")
    parser.add_argument("--image", required=True, help="Path to plate crop image")
    parser.add_argument("--img-width", type=int, default=94)
    parser.add_argument("--img-height", type=int, default=24)
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


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    if args.core_mask == "auto":
        ret = rknn.init_runtime()
    else:
        core_mask = {
            "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,
            "2": RKNNLite.NPU_CORE_2,
        }[args.core_mask]
        ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    inputs = [preprocess(image_path, args.img_width, args.img_height)]
    outputs = rknn.inference(inputs=inputs)
    pred = greedy_decode(outputs[0])
    print(f"[INFO] image={image_path}")
    print(f"[INFO] pred={pred}")

    rknn.release()


if __name__ == "__main__":
    main()
