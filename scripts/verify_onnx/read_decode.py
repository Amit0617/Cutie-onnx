#!/usr/bin/env python3
"""
Verify CUTIE read_decode ONNX graph against PyTorch.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import checker

from scripts.export_onnx_pipeline import (
    CutieReadDecodeForOnnx,
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
    p.add_argument("--memory_len", type=int, default=2)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument(
        "--strict-logits",
        action="store_true",
        help="Fail on logits mismatch as well. By default, parity requires memory_readout/new_sensory/prob only.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    onnx_model = onnx.load(str(args.onnx))
    checker.check_model(onnx_model)
    print("ONNX checker: PASS")

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )

    class Args:
        model = args.model
        weights = args.weights
        disable_object_transformer = False

    model = _load_cutie(Args()).eval()

    b = 1
    n = args.objects
    t = args.memory_len
    h, w = args.height, args.width

    with torch.no_grad():
        image = torch.randn(b, 3, h, w)
        ms, pix = model.encode_image(image)
        key, _, selection = model.transform_key(ms[0])

        h16, w16 = ms[0].shape[-2:]

        sensory = torch.randn(b, n, model.cfg.model.sensory_dim, h16, w16)
        memory_key = torch.randn(b, model.cfg.model.key_dim, t, h16, w16)
        memory_shrinkage = torch.abs(
            torch.randn(b, 1, t, h16, w16)
        ) + 1.0
        memory_mask_value = torch.randn(
            b, n, model.cfg.model.value_dim, t, h16, w16
        )
        object_memory = torch.randn(
            b,
            n,
            t,
            model.cfg.model.object_transformer.num_queries,
            model.cfg.model.embed_dim + 1,
        )
        last_mask = torch.sigmoid(torch.randn(b, n, h16, w16))
        selector = torch.ones(b, n, 1, 1)

        wrapper = CutieReadDecodeForOnnx(model).eval()
        pt_out = wrapper(
            ms[0],
            ms[1],
            ms[2],
            pix,
            key,
            selection,
            memory_key,
            memory_shrinkage,
            memory_mask_value,
            object_memory,
            sensory,
            last_mask,
            selector,
        )

    ort_out = session.run(
        None,
        {
            "ms_feat_16x": ms[0].numpy(),
            "ms_feat_8x": ms[1].numpy(),
            "ms_feat_4x": ms[2].numpy(),
            "pix_feat": pix.numpy(),
            "key": key.numpy(),
            "selection": selection.numpy(),
            "memory_key": memory_key.numpy(),
            "memory_shrinkage": memory_shrinkage.numpy(),
            "memory_mask_value": memory_mask_value.numpy(),
            "object_memory": object_memory.numpy(),
            "sensory": sensory.numpy(),
            "last_mask": last_mask.numpy(),
            "selector": selector.numpy(),
        },
    )

    names = ["memory_readout", "new_sensory", "logits", "prob"]
    required = {"memory_readout", "new_sensory", "prob"}
    if args.strict_logits:
        required.add("logits")

    all_ok = True
    for i, name in enumerate(names):
        a = pt_out[i].numpy()
        b_ = ort_out[i]
        ok = np.allclose(a, b_, rtol=args.rtol, atol=args.atol)
        max_diff = float(np.max(np.abs(a - b_)))
        marker = "required" if name in required else "optional"
        print(f"{name} ({marker}): allclose={ok}, max_diff={max_diff:.6e}")
        if name in required:
            all_ok &= ok

    if not all_ok:
        raise RuntimeError("Numerical mismatch detected in required outputs.")

    print("read_decode parity: PASS")


if __name__ == "__main__":
    main()
