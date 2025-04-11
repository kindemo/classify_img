import os

import cv2
import numpy as np
import tensorflow as tf

def precompute_anchor_indices(config):
    """预处理阶段计算所有可能的bbox尺寸对应的最佳锚框"""
    # 生成所有可能的归一化宽高组合
    width_bins = np.linspace(0, 1, 100)
    height_bins = np.linspace(0, 1, 100)

    anchor_indices = {}
    for w in width_bins:
        for h in height_bins:
            # 计算实际像素尺寸
            pixel_w = w * config.input_size
            pixel_h = h * config.input_size

            # 遍历所有锚框层级
            best_scale = None
            best_anchor = None
            max_iou = -1
            for scale_idx, scale_anchors in enumerate(config.anchors):
                for anchor_idx, (aw, ah) in enumerate(scale_anchors):
                    # 计算IoU
                    intersection = min(pixel_w, aw) * min(pixel_h, ah)
                    union = (pixel_w * pixel_h) + (aw * ah) - intersection
                    iou = intersection / union

                    if iou > max_iou:
                        max_iou = iou
                        best_scale = scale_idx
                        best_anchor = anchor_idx

            anchor_indices[(w, h)] = (best_scale, best_anchor)

    return anchor_indices


def parse_label(label_path, config, anchor_indices):
    labels = {
        'large': np.zeros((13, 13, 3, 6), dtype=np.float32),
        'medium': np.zeros((26, 26, 3, 6), dtype=np.float32),
        'small': np.zeros((52, 52, 3, 6), dtype=np.float32)
    }

    with open(label_path) as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # 使用预计算结果获取锚框索引（性能关键点）
            scale_idx, anchor_idx = anchor_indices.get(
                (width, height), (0, 0)  # 默认值
            )

            # 根据层级填充标签
            grid_size = config.grid_sizes[scale_idx]
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)

            # 计算偏移量
            dx = x_center * grid_size - grid_x
            dy = y_center * grid_size - grid_y

            # 填充对应层级的标签张量
            scale_key = ['large', 'medium', 'small'][scale_idx]
            labels[scale_key][grid_y, grid_x, anchor_idx] = [
                dx, dy,
                np.log(width / config.anchors[scale_idx][anchor_idx][0]),
                np.log(height / config.anchors[scale_idx][anchor_idx][1]),
                1.0,  # 置信度
                class_id
            ]

    return labels


def yolo_generator(config, mode='train'):
    # 动态获取图像路径（不再依赖索引文件）
    image_dir = os.path.join(config.processed_dir, 'images')
    label_dir = os.path.join(config.processed_dir, 'labels')

    # 列出所有图像文件 (假设为png格式)
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # 预计算锚框匹配（保持你的优化逻辑）
    anchors = np.array(config.anchors, dtype=np.float32)
    anchor_indices = precompute_anchor_indices(config)

    for img_file in img_files:
        # 构建完整路径
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))

        # 检查标签文件存在性
        if not os.path.exists(label_path):
            print(f"警告：跳过缺失标签的图像 {img_file}")
            continue

        # 图像加载（保持预处理逻辑）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告：无法加载图像 {img_file}")
            continue

        img = cv2.resize(img, (config.input_size, config.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # 添加通道维度 (H,W,C)

        # 标签解析（保持原有逻辑）
        try:
            labels = parse_label(label_path, config, anchor_indices)
        except Exception as e:
            print(f"解析标签失败: {label_path}, 错误: {str(e)}")
            continue

        yield img, labels


def create_dataset(config, batch_size=16):
    # 修正后的输出签名，每个样本的shape不包含batch维度
    output_signature = (
        tf.TensorSpec(shape=(config.input_size, config.input_size, 1), dtype=tf.float32),
        {
            'large': tf.TensorSpec(shape=(13, 13, 3, 6), dtype=tf.float32),
            'medium': tf.TensorSpec(shape=(26, 26, 3, 6), dtype=tf.float32),
            'small': tf.TensorSpec(shape=(52, 52, 3, 6), dtype=tf.float32)
        }
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: yolo_generator(config),
        output_signature=output_signature
    )

    return dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)