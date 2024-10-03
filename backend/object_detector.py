import logging
import torch
class Detector:
    def __init__(self, model_config):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        model_type = model_config['type']
        model_path = model_config['path']
        # 检查是否有可用的 GPU
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'yolov5':
            from model.yolov5_detector import YOLOv5Detector
            self.model = YOLOv5Detector(model_path)
        elif model_type == 'tensorflow':
            from model.tensorflow_detector import TensorFlowDetector
            self.model = TensorFlowDetector(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"{model_type} model loaded")

    def infer(self, image):
        # 将输入数据加载到相应的设备
        # image = torch.tensor(image).to(self.device)

        results, results_rpn = self.model.infer(image)
        height, width = image.shape[:2]

        normalized_detections = []
        normalized_rpn = []

        for class_label, conf, (x, y, w, h) in results:
            x /= width
            w /= width
            y /= height
            h /= height
            normalized_detections.append((class_label, conf, (x, y, w, h)))

        for class_label, conf, (x, y, w, h) in results_rpn:
            x /= width
            w /= width
            y /= height
            h /= height
            normalized_rpn.append((class_label, conf, (x, y, w, h)))

        return normalized_detections, normalized_rpn