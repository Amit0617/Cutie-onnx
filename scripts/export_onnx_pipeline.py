#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from cutie.model.cutie import CUTIE
from cutie.model.utils.memory_utils import get_affinity, readout


def _weighted_pooling(
    masks: torch.Tensor,
    value: torch.Tensor,
    logits: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    weights = logits.sigmoid() * masks
    sums = torch.einsum("bkhwq,bkhwc->bkqc", weights, value)
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)
    return sums, area


class CutieMemoryWriteForOnnx(torch.nn.Module):
    """
    Memory write/update core:
    encode_mask(image, pix_feat, sensory, masks) -> memory values + updated sensory + object memory.
    """

    def __init__(self, model: CUTIE, h16: int, w16: int):
        super().__init__()
        self.model = model
        self.h16 = h16
        self.w16 = w16

    def forward(
        self,
        image: torch.Tensor,
        pix_feat: torch.Tensor,
        sensory: torch.Tensor,
        masks: torch.Tensor,
    ):
        # Equivalent to CUTIE.encode_mask but with fixed-size mask resizing for ONNX export.
        image = (image - self.model.pixel_mean) / self.model.pixel_std
        others = self.model._get_others(masks)
        msk_value, new_sensory = self.model.mask_encoder(
            image,
            pix_feat,
            sensory,
            masks,
            others,
            deep_update=True,
            chunk_size=-1,
        )

        if self.model.object_transformer_enabled:
            summarizer = self.model.object_summarizer
            masks_fixed = F.interpolate(masks, size=(self.h16, self.w16), mode="area")
            masks_fixed = masks_fixed.unsqueeze(-1)
            inv_masks = 1 - masks_fixed
            repeated_masks = torch.cat(
                [
                    masks_fixed.expand(-1, -1, -1, -1, summarizer.num_summaries // 2),
                    inv_masks.expand(-1, -1, -1, -1, summarizer.num_summaries // 2),
                ],
                dim=-1,
            )

            value = msk_value.permute(0, 1, 3, 4, 2)
            value = summarizer.input_proj(value)
            if summarizer.add_pe:
                value = value + summarizer.pos_enc(value)

            value = value.float()
            feature = summarizer.feature_pred(value)
            logits = summarizer.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)
            obj_memory = torch.cat([sums, area], dim=-1)
        else:
            obj_memory = torch.zeros(
                (masks.shape[0], masks.shape[1], 1, self.model.embed_dim + 1),
                dtype=msk_value.dtype,
                device=msk_value.device,
            )
        return msk_value, new_sensory, obj_memory


class CutieReadDecodeForOnnx(torch.nn.Module):
    """
    Memory read + decode core:
    read_memory(...) + segment(...) -> readout + updated sensory + logits/prob with background.
    """

    def __init__(self, model: CUTIE):
        super().__init__()
        self.model = model

    def forward(
        self,
        ms_feat_16x: torch.Tensor,
        ms_feat_8x: torch.Tensor,
        ms_feat_4x: torch.Tensor,
        pix_feat: torch.Tensor,
        key: torch.Tensor,
        selection: torch.Tensor,
        memory_key: torch.Tensor,
        memory_shrinkage: torch.Tensor,
        memory_mask_value: torch.Tensor,
        object_memory: torch.Tensor,
        sensory: torch.Tensor,
        last_mask: torch.Tensor,
        selector: torch.Tensor,
    ):
        batch_size, num_objects = memory_mask_value.shape[:2]
        with torch.cuda.amp.autocast(enabled=False):
            affinity = get_affinity(
                memory_key.float(),
                memory_shrinkage.float(),
                key.float(),
                selection.float(),
            )
            value = memory_mask_value.flatten(start_dim=1, end_dim=2).float()
            pixel_readout = readout(affinity, value).view(
                batch_size,
                num_objects,
                self.model.value_dim,
                *pix_feat.shape[-2:],
            )
        pixel_readout = self.model.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)
        memory_readout, _ = self.model.readout_query(
            pixel_readout,
            object_memory,
            selector=selector,
        )
        new_sensory, logits, prob = self.model.segment(
            [ms_feat_16x, ms_feat_8x, ms_feat_4x],
            memory_readout,
            sensory,
            selector=selector,
            chunk_size=-1,
            update_sensory=True,
        )
        return memory_readout, new_sensory, logits, prob


def _extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if "model" in loaded and isinstance(loaded["model"], dict):
            return loaded["model"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(loaded)}")
    return loaded


def _toposort_onnx_graph(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    nodes = list(graph.node)

    produced = {}
    for idx, node in enumerate(nodes):
        for out_name in node.output:
            if out_name:
                produced[out_name] = idx

    available: Set[str] = set()
    available.update(v.name for v in graph.input if v.name)
    available.update(v.name for v in graph.initializer if v.name)
    available.update(v.name for v in graph.sparse_initializer if v.name)

    remaining = set(range(len(nodes)))
    sorted_indices: List[int] = []

    while remaining:
        ready = []
        for idx in remaining:
            node = nodes[idx]
            unresolved = False
            for inp in node.input:
                if not inp:
                    continue
                if inp in available:
                    continue
                producer = produced.get(inp)
                if producer is None:
                    continue
                unresolved = True
                break
            if not unresolved:
                ready.append(idx)

        if not ready:
            raise RuntimeError(
                f"Could not topologically sort ONNX graph at {onnx_path}: cycle or unresolved dependency."
            )

        ready.sort()
        for idx in ready:
            remaining.remove(idx)
            sorted_indices.append(idx)
            for out_name in nodes[idx].output:
                if out_name:
                    available.add(out_name)

    sorted_nodes = [nodes[idx] for idx in sorted_indices]
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    onnx.save(model, str(onnx_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CUTIE pipeline ONNX modules (memory_write and read_decode)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to write ONNX files.",
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
    parser.add_argument(
        "--num-objects",
        type=int,
        default=2,
        help="Dummy number of objects used at export time.",
    )
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=4,
        help="Dummy temporal memory length used at export time.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used during export.",
    )
    parser.add_argument(
        "--disable-object-transformer",
        action="store_true",
        help="Disable object transformer (model.object_transformer.num_blocks=0) for exportability.",
    )
    parser.add_argument(
        "--skip-read-decode",
        action="store_true",
        help="Export only memory_write ONNX and skip read_decode export.",
    )
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use torch.onnx dynamo exporter (requires onnxscript).",
    )
    return parser.parse_args()


def _load_cutie(args: argparse.Namespace) -> CUTIE:
    config_dir = (Path(__file__).resolve().parents[1] / "cutie" / "config").as_posix()
    overrides = [f"model={args.model}"]
    if args.disable_object_transformer:
        overrides.append("model.object_transformer.num_blocks=0")
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir, job_name="onnx_export"):
        cfg = compose(config_name="eval_config", overrides=overrides)
    model = CUTIE(cfg)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_weights(_extract_state_dict(checkpoint))
    return model


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("onnx") is None:
        raise ModuleNotFoundError(
            "Missing dependency: onnx. Install it first (e.g. `pip install onnx`) to export."
        )
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.height <= 0 or args.width <= 0 or args.batch <= 0:
        raise ValueError("height/width/batch must be positive integers.")
    if args.num_objects <= 0 or args.memory_frames <= 0:
        raise ValueError("num-objects/memory-frames must be positive integers.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.use_dynamo and args.opset < 18:
        print(f"--use-dynamo requested with opset {args.opset}; overriding to opset 18.")
        args.opset = 18

    device = torch.device(args.device)
    model = _load_cutie(args).to(device).eval()
    read_decode = CutieReadDecodeForOnnx(model).to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_path = args.output_dir / "cutie_memory_write.onnx"
    read_decode_path = args.output_dir / "cutie_read_decode.onnx"

    b = args.batch
    n = args.num_objects
    t = args.memory_frames
    q = model.cfg.model.object_transformer.num_queries
    e_plus_1 = model.cfg.model.embed_dim + 1
    h = args.height
    w = args.width

    with torch.no_grad():
        dummy_image = torch.randn(b, 3, h, w, device=device)
        ms_features, pix_feat = model.encode_image(dummy_image)
        key, _, selection = model.transform_key(ms_features[0])

    h16, w16 = ms_features[0].shape[-2:]
    memory_write = CutieMemoryWriteForOnnx(model, h16, w16).to(device).eval()
    sensory_dim = model.cfg.model.sensory_dim
    value_dim = model.cfg.model.value_dim
    key_dim = model.cfg.model.key_dim

    sensory = torch.randn(b, n, sensory_dim, h16, w16, device=device)
    masks = torch.sigmoid(torch.randn(b, n, h, w, device=device))

    memory_write_kwargs = dict(
        model=memory_write,
        args=(dummy_image, pix_feat, sensory, masks),
        f=str(write_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image", "pix_feat", "sensory", "masks"],
        output_names=["memory_mask_value", "new_sensory", "object_memory"],
        dynamo=args.use_dynamo,
    )
    if not args.use_dynamo:
        memory_write_kwargs["dynamic_axes"] = {
            # NOTE: spatial dims are intentionally static for memory_write.
            # ObjectSummarizer uses adaptive pooling with output size inferred from tensors;
            # legacy ONNX export cannot represent that with dynamic output size.
            "image": {0: "batch"},
            "pix_feat": {0: "batch"},
            "sensory": {0: "batch", 1: "num_objects"},
            "masks": {0: "batch", 1: "num_objects"},
            "memory_mask_value": {
                0: "batch",
                1: "num_objects",
            },
            "new_sensory": {
                0: "batch",
                1: "num_objects",
            },
            "object_memory": {0: "batch", 1: "num_objects"},
        }
    torch.onnx.export(**memory_write_kwargs)
    _toposort_onnx_graph(write_path)

    memory_key = torch.randn(b, key_dim, t, h16, w16, device=device)
    memory_shrinkage = torch.abs(torch.randn(b, 1, t, h16, w16, device=device)) + 1.0
    memory_mask_value = torch.randn(b, n, value_dim, t, h16, w16, device=device)
    object_memory = torch.randn(b, n, t, q, e_plus_1, device=device)
    last_mask = torch.sigmoid(torch.randn(b, n, h16, w16, device=device))
    selector = torch.ones(b, n, 1, 1, device=device)

    if not args.skip_read_decode:
        read_decode_inputs = (
            ms_features[0],
            ms_features[1],
            ms_features[2],
            pix_feat,
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
        read_decode_kwargs = dict(
            model=read_decode,
            args=read_decode_inputs,
            f=str(read_decode_path),
            dynamo=args.use_dynamo,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=[
                "ms_feat_16x",
                "ms_feat_8x",
                "ms_feat_4x",
                "pix_feat",
                "key",
                "selection",
                "memory_key",
                "memory_shrinkage",
                "memory_mask_value",
                "object_memory",
                "sensory",
                "last_mask",
                "selector",
            ],
            output_names=["memory_readout", "new_sensory", "logits", "prob"],
        )
        if not args.use_dynamo:
            read_decode_kwargs["dynamic_axes"] = {
                # NOTE: spatial dims are static for read_decode for exporter compatibility.
                "ms_feat_16x": {0: "batch"},
                "ms_feat_8x": {0: "batch"},
                "ms_feat_4x": {0: "batch"},
                "pix_feat": {0: "batch"},
                "key": {0: "batch"},
                "selection": {0: "batch"},
                "memory_key": {0: "batch", 2: "memory_frames"},
                "memory_shrinkage": {0: "batch", 2: "memory_frames"},
                "memory_mask_value": {0: "batch", 1: "num_objects", 3: "memory_frames"},
                "object_memory": {0: "batch", 1: "num_objects", 2: "memory_frames"},
                "sensory": {0: "batch", 1: "num_objects"},
                "last_mask": {0: "batch", 1: "num_objects"},
                "selector": {0: "batch", 1: "num_objects"},
                "memory_readout": {0: "batch", 1: "num_objects"},
                "new_sensory": {0: "batch", 1: "num_objects"},
                "logits": {0: "batch"},
                "prob": {0: "batch"},
            }
        torch.onnx.export(**read_decode_kwargs)
        _toposort_onnx_graph(read_decode_path)

    print(f"Exported ONNX to: {write_path}")
    if args.skip_read_decode:
        print("Skipped read_decode export.")
    else:
        print(f"Exported ONNX to: {read_decode_path}")


if __name__ == "__main__":
    main()
