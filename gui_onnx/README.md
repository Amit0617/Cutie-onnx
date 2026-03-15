# ONNX GUI Backend

This folder provides an ONNX-backed VOS processor for `interactive_demo_onnx.py` while keeping the same GUI workflow.

## Required ONNX files

- `weights/cutie_image_encoder.onnx`
- `weights/cutie_memory_write.onnx`
- `weights/cutie_read_decode.onnx`
- `weights/ritm_no_brs.onnx`

## Export commands

```bash
python -m scripts.export_onnx --output weights/cutie_image_encoder.onnx --weights weights/cutie-base-mega.pth --height 480 --width 864 --opset 18
python -m scripts.export_onnx_pipeline --weights weights/cutie-base-mega.pth --output-dir weights --use-dynamo --height 480 --width 864 --num-objects 1 --memory-frames 1
python -m scripts.export_ritm_onnx --weights weights/coco_lvis_h18_itermask.pth --output weights/ritm_no_brs.onnx --height 480 --width 864 --opset 18
```

## Run GUI with ONNX backend

```bash
python interactive_demo_onnx.py --images <path_to_images>
python interactive_demo_onnx.py --video <path_to_video>
python interactive_demo_onnx.py <path_to_video>

# ONNX click model (NoBRS)
python interactive_demo_onnx.py --images <path_to_images>
```

Optional custom model paths:

```bash
python interactive_demo_onnx.py --ritm_onnx <ritm_no_brs.onnx> \
  --onnx_encoder <encoder.onnx> \
  --onnx_memory_write <memory_write.onnx> \
  --onnx_read_decode <read_decode.onnx> \
  --images <path_to_images>
```

## Notes

- Current `gui_onnx` backend is single-object oriented and should be run with `--num_objects 1`.
- `gui_onnx` reads limits from the ONNX files at runtime (not hardcoded in code).
- If your current ONNX was exported with small capacities, re-export with larger values:
  - `--num-objects` controls max object count for `memory_write/read_decode`.
  - `--memory-frames` controls temporal memory length accepted by `read_decode`.
- `num_objects` in GUI must be `<=` ONNX-exported object capacity.
