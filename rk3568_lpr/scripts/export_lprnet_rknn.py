from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LPRNet ONNX to RKNN for RK3568.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Path to output .rknn file")
    parser.add_argument("--target", default="rk3568", help="Target Rockchip platform")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--dataset", default="", help="Calibration image txt list for INT8 quantization")
    parser.add_argument("--mean", type=float, default=127.5)
    parser.add_argument("--std", type=float, default=128.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if args.quantize and not args.dataset:
        raise ValueError("--dataset is required when --quantize is enabled")
    if args.quantize and not Path(args.dataset).expanduser().exists():
        raise FileNotFoundError(f"Calibration dataset list not found: {args.dataset}")

    rknn = RKNN(verbose=args.verbose)
    try:
        print("[INFO] configuring RKNN")
        ret = rknn.config(
            target_platform=args.target,
            mean_values=[[args.mean, args.mean, args.mean]],
            std_values=[[args.std, args.std, args.std]],
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        print(f"[INFO] loading ONNX: {onnx_path}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        if args.quantize:
            print(f"[INFO] building INT8 RKNN with dataset: {args.dataset}")
            ret = rknn.build(do_quantization=True, dataset=str(Path(args.dataset).expanduser().resolve()))
        else:
            print("[INFO] building FP16 RKNN without quantization")
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        print(f"[INFO] exporting RKNN: {output_path}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")

        print(f"[INFO] done: {output_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
