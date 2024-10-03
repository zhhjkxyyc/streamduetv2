# model/yolov5_detector.py
import torch
import logging
import os
import numpy as np
from PIL import Image, ImageDraw

class YOLOv5Detector:
    classes = {
        "person": [0],
        "bicycle": [1],
        "car": [2,3, 6, 8],
        # "person": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }
    rpn_threshold = 0.1
    current_image_id = 0  # Class variable to track image IDs

    def __init__(self, model_path):
        self.logger = logging.getLogger("yolov5_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.model = None
        self.load_model(model_path)
        self.output_dir = "results/output_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_model(self, model_path):
        self.logger.info(f"Loading YOLOv5 model from {model_path}")

        # 加载本地模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        self.model.eval()  # 设置模型为推理模式

    def get_class_label(self, label):
        for class_name, labels in self.classes.items():
            if label in labels:
                return class_name
        return "object"

    def infer(self, image):
        # Perform inference on the image
        results = self.model(image)
        detection_results = results.xywh[0].cpu().numpy()
        # Convert results to the required format
        detections = []
        results_rpn = []
        for x, y, w, h, conf, label in detection_results:
            x = x - w / 2
            y = y - h / 2
            class_label = self.get_class_label(int(label))
            if class_label == "object" and conf < YOLOv5Detector.rpn_threshold:
                results_rpn.append((class_label, float(conf), (float(x), float(y), float(w), float(h))))
                continue
            detections.append((class_label, float(conf), (float(x), float(y), float(w), float(h))))

        # Draw results on images and save
        # image_id = YOLOv5Detector.current_image_id
        # YOLOv5Detector.current_image_id += 1
        #
        # self.draw_and_save(image, detections, os.path.join(self.output_dir, f"detections_{image_id}.png"), f"Detections {image_id}")
        # self.draw_and_save(image, results_rpn, os.path.join(self.output_dir, f"rpn_results_{image_id}.png"), f"RPN Results {image_id}")

        return detections, results_rpn

    def draw_and_save(self, image, results, output_path, title):
        # Convert tensor or numpy array image to PIL Image
        if isinstance(image, torch.Tensor):
            image = image.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)

        draw = ImageDraw.Draw(image)
        for class_label, conf, (x, y, w, h) in results:
            left = x
            top = y
            right = x + w / 2
            bottom = y + h / 2
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            draw.text((left, top), f"{class_label} {conf:.2f}", fill="red")

        # Save image
        image.save(output_path)
        self.logger.info(f"{title} saved to {output_path}")
