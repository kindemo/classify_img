# config.py - 配置文件
from src.yolo_mini import ANCHORS


class Config:
    # 预处理输出配置
    PREPROCESS = {
        "output_dir": "./data/processed",
        "image_size": (416, 416),
        "classes": ["nodule_1", "nodule_2", "nodule_3", "nodule_4", "nodule_5"],
        "annotation_format": "yolo"  # 支持yolo/coco格式
    }

    # 模型配置
    MODEL = {
        "input_shape": (416, 416, 1),
        "anchors": ANCHORS,  # 使用之前定义的锚框
        "num_classes": 5
    }


config = Config()