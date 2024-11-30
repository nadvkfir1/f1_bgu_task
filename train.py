"""
Author: Nadav Kfir

Train AI model from traffic cone detection
"""


from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

model.train(
    data=Path(__file__).parent / "resources" / "data.yaml",
    name="traffic_cone_model",
)