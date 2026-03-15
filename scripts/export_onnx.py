#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from cutie.model.cutie import CUTIE


class CutieImageEncoderForOnnx(torch.nn.Module):
    def __init__(self, model: CUTIE):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        ms_features, pix_feat = self.model.encode_image(image)
        key, shrinkage, selection = self.model.transform_key(ms_features[0])
        return (
            ms_features[0],
            ms_features[1],
            ms_features[2],
            pix_feat,
            key,
            shrinkage,
            selection,
        )


def _extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if "model" in loaded and isinstance(loaded["model"], dict):
            return loaded["model"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(loaded)}")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CUTIE image encoder path to ONNX (encode_image + transform_key)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX file path.",
    )
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
    parser.add_argument("--height", type=int, default=480, help="Dummy input height for export.")
    parser.add_argument("--width", type=int, default=854, help="Dummy input width for export.")
    parser.add_argument("--batch", type=int, default=1, help="Dummy input batch size for export.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used during export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("onnx") is None:
        raise ModuleNotFoundError(
            "Missing dependency: onnx. Install it first (e.g. `pip install onnx`) to export."
        )
    if args.height <= 0 or args.width <= 0 or args.batch <= 0:
        raise ValueError("height/width/batch must be positive integers.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    config_dir = (Path(__file__).resolve().parents[1] / "cutie" / "config").as_posix()
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir, job_name="onnx_export"):
        cfg = compose(config_name="eval_config", overrides=[f"model={args.model}"])
    with open_dict(cfg):
        cfg.weights = str(args.weights)

    model = CUTIE(cfg)
    if args.weights.exists():
        checkpoint = torch.load(args.weights, map_location="cpu")
        model.load_weights(_extract_state_dict(checkpoint))
    else:
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = torch.device(args.device)
    model = model.to(device).eval()
    wrapper = CutieImageEncoderForOnnx(model).to(device).eval()

    dummy = torch.randn(args.batch, 3, args.height, args.width, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.output),
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=[
            "ms_feat_16x",
            "ms_feat_8x",
            "ms_feat_4x",
            "pix_feat",
            "key",
            "shrinkage",
            "selection",
        ],
        dynamic_axes={
            "image": {
                0: "batch",
                2: "height",
                3: "width",
            },
            "ms_feat_16x": {
                0: "batch",
                2: "height_div16",
                3: "width_div16",
            },
            "ms_feat_8x": {
                0: "batch",
                2: "height_div8",
                3: "width_div8",
            },
            "ms_feat_4x": {
                0: "batch",
                2: "height_div4",
                3: "width_div4",
            },
            "pix_feat": {
                0: "batch",
                2: "height_div16",
                3: "width_div16",
            },
            "key": {
                0: "batch",
                2: "height_div16",
                3: "width_div16",
            },
            "shrinkage": {
                0: "batch",
                2: "height_div16",
                3: "width_div16",
            },
            "selection": {
                0: "batch",
                2: "height_div16",
                3: "width_div16",
            },
        },
    )

    print(f"Exported ONNX to: {args.output}")


if __name__ == "__main__":
    main()
