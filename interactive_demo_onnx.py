import logging
import os
import signal
import sys
from argparse import ArgumentParser
from hashlib import sha1
from pathlib import Path

if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

signal.signal(signal.SIGINT, signal.SIG_DFL)


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--images", help="Folders containing input images.", default=None)
    parser.add_argument("--video", help="Video file readable by OpenCV.", default=None)
    parser.add_argument(
        "--workspace",
        help="directory for storing buffered images (if needed) and output masks",
        default=None,
    )
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument(
        "--workspace_init_only",
        action="store_true",
        help="initialize the workspace and exit",
    )
    parser.add_argument(
        "--onnx_encoder",
        default="weights/cutie_image_encoder.onnx",
        help="path to ONNX image encoder",
    )
    parser.add_argument(
        "--onnx_memory_write",
        default="weights/cutie_memory_write.onnx",
        help="path to ONNX memory_write module",
    )
    parser.add_argument(
        "--onnx_read_decode",
        default="weights/cutie_read_decode.onnx",
        help="path to ONNX read_decode module",
    )
    parser.add_argument(
        "--ritm_onnx",
        default="weights/ritm_no_brs.onnx",
        help="path to ONNX RITM click model",
    )
    parser.add_argument(
        "--click_backend_model",
        choices=["ritm", "sam2"],
        default="sam2",
        help="first-frame click backend: RITM or SAM2",
    )
    parser.add_argument(
        "--sam2_encoder_onnx",
        default="weights/sam2.1_hiera_small.encoder.onnx",
        help="path to ONNX SAM2 encoder model used for click prompts",
    )
    parser.add_argument(
        "--sam2_decoder_onnx",
        default="weights/sam2.1_hiera_small.decoder.onnx",
        help="path to ONNX SAM2 decoder model used for click prompts",
    )
    parser.add_argument(
        "--ritm_max_clicks",
        type=int,
        default=8,
        help="max click history used to build ONNX click maps",
    )
    parser.add_argument(
        "--ritm_click_radius",
        type=int,
        default=5,
        help="disk radius for ONNX click maps",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ONNX Runtime execution provider",
    )
    argv = sys.argv[1:]
    if len(argv) > 0 and not argv[0].startswith("-"):
        argv = ["--video", argv[0], *argv[1:]]
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    import onnxruntime as ort

    if requested != "auto":
        return requested

    providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    return "cpu"


def resolve_config_dir() -> Path:
    base_dir = Path(__file__).resolve().parent
    config_dir = base_dir / "cutie" / "config"
    if config_dir.exists():
        return config_dir
    return Path("cutie/config")


def resolve_runtime_path(raw_path: str) -> str:
    path_obj = Path(raw_path)
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)

    base_dir = Path(__file__).resolve().parent
    bundled_path = base_dir / raw_path
    if bundled_path.exists():
        return str(bundled_path)

    return raw_path


def default_workspace_root() -> Path:
    return Path.home() / "scutie"


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_workspace(args) -> str | None:
    if args.workspace is not None:
        return args.workspace

    workspace_root = default_workspace_root()
    workspace_root.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        return str(workspace_root / sha1_file(video_path))

    if args.images is not None:
        images_path = Path(args.images).expanduser().resolve()
        return str(workspace_root / images_path.name)

    return None


if __name__ in "__main__":
    args = get_arguments()

    from hydra import compose, initialize, initialize_config_dir
    from omegaconf import open_dict
    from PySide6.QtWidgets import QApplication
    import qdarktheme

    from gui_onnx.main_controller import MainControllerOnnxNumpy

    log = logging.getLogger()

    config_dir = resolve_config_dir()
    if config_dir.is_absolute():
        with initialize_config_dir(
            version_base="1.3.2",
            config_dir=str(config_dir),
            job_name="gui_onnx",
        ):
            cfg = compose(config_name="gui_config")
    else:
        with initialize(
            version_base="1.3.2",
            config_path=config_dir.as_posix(),
            job_name="gui_onnx",
        ):
            cfg = compose(config_name="gui_config")

    args.device = resolve_device(args.device)
    log.info(f"Using ONNX Runtime device: {args.device}")
    args.video = resolve_runtime_path(args.video) if args.video is not None else None
    args.images = resolve_runtime_path(args.images) if args.images is not None else None
    args.workspace = resolve_workspace(args)

    args_dict = vars(args)
    for key in (
        "onnx_encoder",
        "onnx_memory_write",
        "onnx_read_decode",
        "ritm_onnx",
        "sam2_encoder_onnx",
        "sam2_decoder_onnx",
    ):
        args_dict[key] = resolve_runtime_path(args_dict[key])

    with open_dict(cfg):
        for key, value in args_dict.items():
            assert key not in cfg, f"Argument {key} already exists in config"
            cfg[key] = value
        cfg["backend"] = "onnx"
        cfg["click_backend"] = "onnx"
        cfg["amp"] = False

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    ex = MainControllerOnnxNumpy(cfg)
    if "workspace_init_only" in cfg and cfg["workspace_init_only"]:
        sys.exit(0)
    sys.exit(app.exec())
