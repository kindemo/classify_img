import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import config

class YOLOLabelGenerator:
    def __init__(self, config):
        self.config = config
        self.class_id = 1  # 结节类别ID
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
        """生成单个尺度的标签文件"""
        grid_size = self.config.grid_sizes[scale]
        anchors = self.config.anchors[scale]

        x_center = annot['x_center']
        y_center = annot['y_center']
        width = annot['width']
        height = annot['height']

        # 验证坐标范围
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            print(f"警告：异常坐标值 {x_center}, {y_center}")
            return

        # 计算网格位置和偏移
        grid_x = int(x_center * grid_size)
        grid_y = int(y_center * grid_size)
        tx = x_center * grid_size - grid_x
        ty = y_center * grid_size - grid_y

        # 匹配最佳锚框
        best_anchor = self._find_best_anchor(width, height, anchors)

        # 计算尺寸参数
        tw = np.log(width / anchors[best_anchor][0] + 1e-8)
        th = np.log(height / anchors[best_anchor][1] + 1e-8)

        # 构建标签行（包含scale, grid_x, grid_y, anchor_idx, tx, ty, tw, th, conf=1, class_id）
        label_line = f"{scale} {grid_x} {grid_y} {best_anchor} {tx:.4f} {ty:.4f} {tw:.4f} {th:.4f} 1 {self.class_id}\n"

        # 保存标签
        self._save_label(img_path, scale, label_line)

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