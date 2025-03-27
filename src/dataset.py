# dataset.py - 数据加载器 实现数据加载管道
from glob import glob

import numpy as np
import tensorflow as tf


class YoloDataset:
    def __init__(self, config):
        self.image_size = config.MODEL["input_shape"][:2]
        self.class_map = {cls: idx for idx, cls in enumerate(config.PREPROCESS["classes"])}

    def _parse_annotation(self, label_path):
        """解析YOLO格式标注"""
        with open(label_path) as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            class_id, xc, yc, w, h = map(float, line.strip().split())
            boxes.append([xc, yc, w, h, int(class_id)])
        return np.array(boxes)

    def load_dataset(self, data_dir):
        """生成TF Dataset"""
        image_paths = glob(f"{data_dir}/images/*.png")
        dataset = tf.data.Dataset.from_generator(
            self._data_generator(image_paths),
            output_signature=(
                tf.TensorSpec(shape=(*self.image_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
            )
        )
        return dataset

    def _data_generator(self, image_paths):
        def gen():
            for img_path in image_paths:
                # 加载图像
                image = tf.io.read_file(img_path)
                image = tf.image.decode_png(image, channels=1)
                image = tf.image.resize(image, self.image_size)

                # 加载标注
                label_path = img_path.replace("images", "labels").replace(".png", ".txt")
                boxes = self._parse_annotation(label_path)

                yield image / 255.0, boxes

        return gen