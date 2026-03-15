#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import onnx
import torch
from hydra import compose, initialize_config_dir
from onnx import checker, shape_inference
from onnx.reference import ReferenceEvaluator

from cutie.model.cutie import CUTIE
from scripts.export_onnx import CutieImageEncoderForOnnx, _extract_state_dict

OUTPUT_NAMES: Sequence[str] = (
    "ms_feat_16x",
    "ms_feat_8x",
    "ms_feat_4x",
    "pix_feat",
    "key",
    "shrinkage",
    "selection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CUTIE ONNX (model check + inference parity vs PyTorch)."
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model file.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/cutie-base-mega.pth"),
        help="Checkpoint path for CUTIE weights.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "small"],
        help="CUTIE model config variant.",
    )
    parser.add_argument("--height", type=int, default=128, help="Validation input height.")
    parser.add_argument("--width", type=int, default=128, help="Validation input width.")
    parser.add_argument("--batch", type=int, default=1, help="Validation input batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="allclose rtol.")
    parser.add_argument("--atol", type=float, default=1e-4, help="allclose atol.")
    return parser.parse_args()


def _load_pytorch_wrapper(model_name: str, weights_path: Path) -> CutieImageEncoderForOnnx:
    config_dir = (Path(__file__).resolve().parents[1] / "cutie" / "config").as_posix()
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir, job_name="onnx_verify"):
        cfg = compose(config_name="eval_config", overrides=[f"model={model_name}"])
    model = CUTIE(cfg)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_weights(_extract_state_dict(checkpoint))
    return CutieImageEncoderForOnnx(model).eval()


def _run_onnx(model: onnx.ModelProto, image: np.ndarray):
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, {"image": image}), "onnxruntime"
    return ReferenceEvaluator(model).run(None, {"image": image}), "onnx.reference"


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if args.height <= 0 or args.width <= 0 or args.batch <= 0:
        raise ValueError("height/width/batch must be positive integers.")

    onnx_model = onnx.load(str(args.onnx))
    checker.check_model(onnx_model)
    inferred = shape_inference.infer_shapes(onnx_model)
    print("ONNX checker: PASS")
    print("ONNX outputs:", [o.name for o in inferred.graph.output])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    input_tensor = torch.randn(args.batch, 3, args.height, args.width, dtype=torch.float32)

    start = time.time()
    wrapper = _load_pytorch_wrapper(args.model, args.weights)
    with torch.no_grad():
        torch_out = wrapper(input_tensor)
        print("PyTorch inference time:", time.time() - start, "seconds")

    start = time.time()
    onnx_out, engine = _run_onnx(onnx_model, input_tensor.numpy())
    print("ONNX inference time:", time.time() - start, "seconds")
    print(f"ONNX runtime engine: {engine}")

    all_ok = True
    for i, name in enumerate(OUTPUT_NAMES):
        pt = torch_out[i].detach().cpu().numpy()
        ox = onnx_out[i]
        ok = np.allclose(pt, ox, rtol=args.rtol, atol=args.atol)
        max_abs = float(np.max(np.abs(pt - ox)))
        mean_abs = float(np.mean(np.abs(pt - ox)))
        print(
            f"{name}: shape={pt.shape}, allclose={ok}, "
            f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}"
        )
        all_ok = all_ok and ok

    if not all_ok:
        raise RuntimeError("ONNX parity check failed.")
    print("Numerical check: PASS")


if __name__ == "__main__":
    main()
