#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from gui.ritm.inference import utils as ritm_utils
import gui.ritm.model.is_hrnet_model  # noqa: F401 - required for config class resolution


class RITMNoBRSForOnnx(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, coord_features: torch.Tensor) -> torch.Tensor:
        # image: Bx4xHxW (RGB + prev mask)
        # coord_features: Bx2xHxW (pos/neg click maps)
        rgb, prev_mask = self.model.prepare_input(image)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        if self.model.rgb_conv is not None:
            x = self.model.rgb_conv(torch.cat((rgb, coord_features), dim=1))
            outputs = self.model.backbone_forward(x)
        else:
            coord_features = self.model.maps_transform(coord_features)
            outputs = self.model.backbone_forward(rgb, coord_features)
        return F.interpolate(
            outputs["instances"],
            size=rgb.size()[2:],
            mode="bilinear",
            align_corners=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export RITM click model (NoBRS forward) to ONNX."
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/coco_lvis_h18_itermask.pth"),
        help="RITM checkpoint path.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("weights/ritm_no_brs.onnx"),
        help="Output ONNX path.",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    # Use torch-based distance maps for ONNX exportability (cpu_dist_maps=True uses numpy/cython path).
    model = ritm_utils.load_is_model(str(args.weights), device="cpu", cpu_dist_maps=False).eval()
    wrapper = RITMNoBRSForOnnx(model).eval()

    # with_prev_mask=True for the shipped checkpoint; image has 4 channels (RGB + prev mask).
    image = torch.randn(1, 4, args.height, args.width)
    coord_features = torch.zeros(1, 2, args.height, args.width, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (image, coord_features),
        str(args.output),
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image", "coord_features"],
        output_names=["instances"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "coord_features": {2: "height", 3: "width"},
            "instances": {2: "height", 3: "width"},
        },
    )

    # Validate exported graph still exposes click points as an input.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    inputs = [x.name for x in sess.get_inputs()]
    if "coord_features" not in inputs:
        raise RuntimeError(
            f"Exported model {args.output} does not include 'coord_features' input (inputs: {inputs}). "
            "The graph is not usable for interactive clicks."
        )
    print(f"Exported ONNX to: {args.output}")


if __name__ == "__main__":
    main()
import torch.nn.functional as F
