import argparse
import os
from pathlib import Path


def _resolve_data_yaml(data: str | None) -> Path:
	if data:
		p = Path(data).expanduser()
	else:
		# Default to your dataset location in this repo
		p = (
			Path(__file__).resolve().parents[1]
			/ "dataset"
			/ "DIP_coursedesign.v2i.yolov8"
			/ "data.yaml"
		)
	return p.resolve()


def _validate_paths(data_yaml: Path) -> None:
	if not data_yaml.exists():
		raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

	# Optional: sanity-check that referenced image folders exist
	try:
		import yaml
	except Exception:
		# If PyYAML isn't installed yet, skip deeper validation.
		return

	cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
	base = data_yaml.parent
	for key in ("train", "val", "test"):
		v = cfg.get(key)
		if isinstance(v, str):
			img_dir = (base / v).resolve()
			if not img_dir.exists():
				raise FileNotFoundError(f"{key} path not found: {img_dir} (from {v} in {data_yaml})")


def train(args: argparse.Namespace) -> None:
	from ultralytics import YOLO

	data_yaml = _resolve_data_yaml(args.data)
	_validate_paths(data_yaml)

	model = YOLO(args.model)
	model.train(
		data=str(data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		project=args.project,
		name=args.name,
		exist_ok=args.exist_ok,
		pretrained=args.pretrained,
		patience=args.patience,
		cos_lr=args.cos_lr,
	)


def validate(args: argparse.Namespace) -> None:
	from ultralytics import YOLO

	data_yaml = _resolve_data_yaml(args.data)
	_validate_paths(data_yaml)

	model = YOLO(args.weights)
	model.val(
		data=str(data_yaml),
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		split=args.split,
	)


def predict(args: argparse.Namespace) -> None:
	from ultralytics import YOLO

	model = YOLO(args.weights)
	model.predict(
		source=args.source,
		imgsz=args.imgsz,
		conf=args.conf,
		iou=args.iou,
		device=args.device,
		save=True,
		project=args.project,
		name=args.name,
		exist_ok=args.exist_ok,
	)


def export_model(args: argparse.Namespace) -> None:
	from ultralytics import YOLO

	model = YOLO(args.weights)
	model.export(
		format=args.format,
		imgsz=args.imgsz,
		device=args.device,
		half=args.half,
		dynamic=args.dynamic,
		simplify=args.simplify,
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Train/Val/Predict YOLOv8 using DIP_coursedesign dataset (local data.yaml)."
	)
	sub = parser.add_subparsers(dest="cmd", required=True)

	p_train = sub.add_parser("train", help="Train YOLOv8")
	p_train.add_argument(
		"--data",
		type=str,
		default=None,
		help="Path to data.yaml (default: dataset/DIP_coursedesign.v2i.yolov8/data.yaml)",
	)
	p_train.add_argument(
		"--model",
		type=str,
		default="yolov8n.pt",
		help="Base model or weights (e.g., yolov8n.pt / yolov8s.pt / runs/.../best.pt)",
	)
	p_train.add_argument("--epochs", type=int, default=100)
	p_train.add_argument("--imgsz", type=int, default=640)
	p_train.add_argument("--batch", type=int, default=16)
	p_train.add_argument("--device", type=str, default="0", help="'0' for GPU0, 'cpu' for CPU")
	p_train.add_argument("--workers", type=int, default=8)
	p_train.add_argument("--project", type=str, default="runs/train")
	p_train.add_argument("--name", type=str, default="dip_cone")
	p_train.add_argument("--exist-ok", action="store_true")
	p_train.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
	p_train.add_argument("--patience", type=int, default=50)
	p_train.add_argument("--cos-lr", action="store_true")
	p_train.set_defaults(func=train)

	p_val = sub.add_parser("val", help="Validate a trained model")
	p_val.add_argument(
		"--data",
		type=str,
		default=None,
		help="Path to data.yaml (default: dataset/DIP_coursedesign.v2i.yolov8/data.yaml)",
	)
	p_val.add_argument("--weights", type=str, required=True, help="Path to weights (best.pt/last.pt) or .pt model")
	p_val.add_argument("--imgsz", type=int, default=640)
	p_val.add_argument("--batch", type=int, default=16)
	p_val.add_argument("--device", type=str, default="0")
	p_val.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
	p_val.set_defaults(func=validate)

	p_pred = sub.add_parser("predict", help="Run inference on images/videos")
	p_pred.add_argument("--weights", type=str, required=True)
	p_pred.add_argument("--source", type=str, required=True, help="Image/dir/video/webcam (e.g., 0)")
	p_pred.add_argument("--imgsz", type=int, default=640)
	p_pred.add_argument("--conf", type=float, default=0.25)
	p_pred.add_argument("--iou", type=float, default=0.7)
	p_pred.add_argument("--device", type=str, default="0")
	p_pred.add_argument("--project", type=str, default="runs/predict")
	p_pred.add_argument("--name", type=str, default="dip_cone")
	p_pred.add_argument("--exist-ok", action="store_true")
	p_pred.set_defaults(func=predict)

	p_exp = sub.add_parser("export", help="Export model to ONNX/TensorRT/etc")
	p_exp.add_argument("--weights", type=str, required=True)
	p_exp.add_argument("--format", type=str, default="onnx")
	p_exp.add_argument("--imgsz", type=int, default=640)
	p_exp.add_argument("--device", type=str, default="cpu")
	p_exp.add_argument("--half", action="store_true")
	p_exp.add_argument("--dynamic", action="store_true")
	p_exp.add_argument("--simplify", action="store_true")
	p_exp.set_defaults(func=export_model)

	return parser


def main() -> None:
	# Reduce OpenMP thread oversubscription on some Linux setups
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
