# course_design (YOLOv8)

使用本仓库内的标注数据集 `dataset/DIP_coursedesign.v2i.yolov8` 训练 YOLOv8。

## 安装依赖

```bash
cd /home/serenade/Documents/course_design
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 训练

默认会读取：`dataset/DIP_coursedesign.v2i.yolov8/data.yaml`

```bash
python src/yolo.py train --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

如果只想用 CPU：

```bash
python src/yolo.py train --device cpu
```

## 验证

```bash
python src/yolo.py val --weights runs/train/dip_cone/weights/best.pt --device 0
```

## 推理

```bash
python src/yolo.py predict --weights runs/train/dip_cone/weights/best.pt --source dataset/DIP_coursedesign.v2i.yolov8/test/images --device 0
```

## 导出（ONNX）

```bash
python src/yolo.py export --weights runs/train/dip_cone/weights/best.pt --format onnx
```
