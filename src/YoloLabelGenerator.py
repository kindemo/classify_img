import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import config

class YOLOLabelGenerator:
    def __init__(self, config):
        self.config = config
        self.class_id = 1   # 代表结节类别ID
        Path(config.label_dir).mkdir(parents=True, exist_ok=True)

    def generate_labels(self, excel_path):
        df = pd.read_excel(excel_path)
        required_columns = ['seriesuid', 'x_center', 'y_center', 'width', 'height']  # 移除z_index
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Excel缺少必要列: {required_columns}")

        for _, row in df.iterrows():
            # 生成标准2D文件名
            img_name = f"{row['seriesuid'].strip()}.png"
            img_path = os.path.join(self.config.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告：图片文件 {img_name} 不存在，跳过生成标签")
                continue
            for scale in range(len(self.config.grid_sizes)):
                self.create_scale_label(img_path, row, scale)

    def create_scale_label(self, img_path, annot, scale):
        """生成符合YOLO标准的标签"""
        x_center = annot['x_center']
        y_center = annot['y_center']
        width = annot['width']
        height = annot['height']

        label_line = f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        # 保存标签（所有尺度共享同一个标签文件）
        label_path = Path(img_path).with_suffix('.txt')
        with open(label_path, "a") as f:
            f.write(label_line)

    @staticmethod
    def _find_best_anchor(width, height, anchors):
        """基于IoU选择最佳锚框"""
        best_iou = -1
        best_idx = 0
        for i, (aw, ah) in enumerate(anchors):
            min_w = min(width, aw)
            min_h = min(height, ah)
            intersection = min_w * min_h
            union = (width * height) + (aw * ah) - intersection
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx

    def _save_label(self, img_path, scale, label_line):
        """保存标签文件，文件名与图像对应"""
        seriesuid = Path(img_path).stem  # 获取不带扩展名的文件名
        label_name = f"{seriesuid}.txt"  # 标签文件名与图像同名，扩展名为.txt
        label_path = os.path.join(self.config.label_dir, label_name)

        # 覆盖写入模式
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label_line)


# 使用示例
if __name__ == "__main__":
    generator = YOLOLabelGenerator(config)
    # 从包含归一化坐标的Excel生成标签
    generator.generate_labels(config.output_excel_path)