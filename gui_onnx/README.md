# ONNX GUI Backend

This folder provides an ONNX-backed VOS processor for `interactive_demo_onnx.py` while keeping the same GUI workflow.

## Required ONNX files

- `weights/cutie_image_encoder.onnx`
- `weights/cutie_memory_write.onnx`
- `weights/cutie_read_decode.onnx`
- `weights/ritm_no_brs.onnx`
- Optional for SAM2-assisted first-frame clicks:
  - `weights/sam2.1_hiera_small.encoder.onnx`
  - `weights/sam2.1_hiera_small.decoder.onnx`

## Export commands

```bash
python -m scripts.export_onnx --output weights/cutie_image_encoder.onnx --weights weights/cutie-base-mega.pth --height 480 --width 864 --opset 18
python -m scripts.export_onnx_pipeline --weights weights/cutie-base-mega.pth --output-dir weights --use-dynamo --height 480 --width 864 --num-objects 1 --memory-frames 1
python -m scripts.export_ritm_onnx --weights weights/coco_lvis_h18_itermask.pth --output weights/ritm_no_brs.onnx --height 480 --width 864 --opset 18
python -m samexporter.export_sam2 --checkpoint weights/sam2.1_hiera_small.pt --output_encoder weights/sam2.1_hiera_small.encoder.onnx --output_decoder weights/sam2.1_hiera_small.decoder.onnx --model_type sam2.1_hiera_small
```

## Run GUI with ONNX backend

```bash
python interactive_demo_onnx.py --images <path_to_images>
python interactive_demo_onnx.py --video <path_to_video>
python interactive_demo_onnx.py <path_to_video>

# ONNX click model (NoBRS)
python interactive_demo_onnx.py --images <path_to_images>

# SAM2 clicks for first-frame object selection, Cutie ONNX for propagation
python interactive_demo_onnx.py \
  --images <path_to_images> \
  --click_backend_model sam2

# In the GUI:
# - left click: positive point
# - right click: negative point
# - shift + left-drag: rectangle box prompt for SAM2
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
- With `--click_backend_model sam2`, clicks only affect first-frame mask generation; Cutie still handles all temporal tracking/memory.
- The GitHub Actions release workflow exports and bundles the SAM2.1 small encoder/decoder into `weights/`.
- If your current ONNX was exported with small capacities, re-export with larger values:
  - `--num-objects` controls max object count for `memory_write/read_decode`.
  - `--memory-frames` controls temporal memory length accepted by `read_decode`.
- `num_objects` in GUI must be `<=` ONNX-exported object capacity.
