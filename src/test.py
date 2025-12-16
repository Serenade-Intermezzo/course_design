import argparse
import time
from pathlib import Path

import cv2


def _default_weights() -> str:
	repo_root = Path(__file__).resolve().parents[1]
	best = repo_root / "runs" / "train" / "dip_cone" / "weights" / "best.pt"
	return str(best) if best.exists() else "yolov8n.pt"


def _auto_device(requested: str | None) -> str:
	if requested:
		return requested
	try:
		import torch

		return "0" if torch.cuda.is_available() else "cpu"
	except Exception:
		return "cpu"


def run(args: argparse.Namespace) -> None:
	from ultralytics import YOLO

	device = _auto_device(args.device)
	model = YOLO(args.weights)

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		raise RuntimeError(f"无法打开摄像头：{args.camera}（检查是否被占用/权限/设备号）")

	if args.width:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	if args.height:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

	win = args.window
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)

	last_t = time.time()
	fps = 0.0

	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			break

		# Ultralytics Results: result.plot() 会把框/标签画到图上
		results = model.predict(
			source=frame,
			imgsz=args.imgsz,
			conf=args.conf,
			iou=args.iou,
			device=device,
			verbose=False,
		)
		annotated = results[0].plot()

		# FPS 计算
		now = time.time()
		dt = now - last_t
		last_t = now
		if dt > 0:
			fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

		cv2.putText(
			annotated,
			f"device={device}  FPS={fps:.1f}  conf>={args.conf}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)

		cv2.imshow(win, annotated)
		key = cv2.waitKey(1) & 0xFF
		if key in (ord("q"), 27):  # q 或 ESC 退出
			break

	cap.release()
	cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Webcam realtime detection with YOLOv8 (CUDA supported).")
	p.add_argument("--weights", type=str, default=_default_weights(), help="模型权重：best.pt 或 yolov8n.pt")
	p.add_argument("--camera", type=int, default=0, help="摄像头设备号（默认 0）")
	p.add_argument("--imgsz", type=int, default=640)
	p.add_argument("--conf", type=float, default=0.25)
	p.add_argument("--iou", type=float, default=0.7)
	p.add_argument(
		"--device",
		type=str,
		default=None,
		help="推理设备：'0' 使用 GPU0，'cpu' 使用 CPU；不填则自动优先 CUDA",
	)
	p.add_argument("--width", type=int, default=0, help="可选：设置摄像头输出宽度")
	p.add_argument("--height", type=int, default=0, help="可选：设置摄像头输出高度")
	p.add_argument("--window", type=str, default="YOLO Webcam")
	return p


def main() -> None:
	args = build_parser().parse_args()
	run(args)


if __name__ == "__main__":
	main()
