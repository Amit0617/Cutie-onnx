"""
Verify CUTIE memory_write ONNX graph against PyTorch.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import checker

from scripts.export_onnx_pipeline import (
    CutieMemoryWriteForOnnx,
    _load_cutie,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--model", type=str, default="base")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--objects", type=int, default=2)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()

    # Load ONNX
    onnx_model = onnx.load(str(args.onnx))
    checker.check_model(onnx_model)
    print("ONNX checker: PASS")

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )

    # Load CUTIE
    class Args:
        model = args.model
        weights = args.weights
        disable_object_transformer = False

    model = _load_cutie(Args()).eval()

    b = 1
    n = args.objects
    h, w = args.height, args.width

    with torch.no_grad():
        image = torch.randn(b, 3, h, w)
        ms, pix = model.encode_image(image)

        h16, w16 = ms[0].shape[-2:]

        sensory = torch.randn(
            b, n, model.cfg.model.sensory_dim, h16, w16
        )
        masks = torch.sigmoid(torch.randn(b, n, h, w))

        wrapper = CutieMemoryWriteForOnnx(model, h16, w16).eval()
        pt_out = wrapper(image, pix, sensory, masks)

    ort_out = session.run(
        None,
        {
            "image": image.numpy(),
            "pix_feat": pix.numpy(),
            "sensory": sensory.numpy(),
            "masks": masks.numpy(),
        },
    )

    names = ["memory_mask_value", "new_sensory", "object_memory"]

    all_ok = True
    for i, name in enumerate(names):
        a = pt_out[i].numpy()
        b_ = ort_out[i]
        ok = np.allclose(a, b_, rtol=args.rtol, atol=args.atol)
        max_diff = float(np.max(np.abs(a - b_)))
        print(f"{name}: allclose={ok}, max_diff={max_diff:.6e}")
        all_ok &= ok

    if not all_ok:
        raise RuntimeError("Numerical mismatch detected.")

    print("memory_write parity: PASS")


if __name__ == "__main__":
    main()