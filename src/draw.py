import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.yolo_mini import ANCHORS


def decode_predictions(preds, anchors, num_classes, conf_threshold=0.5, iou_threshold=0.4):
    """将模型输出解码为边界框"""
    boxes = []
    scores = []
    class_ids = []

    # 遍历三个检测尺度
    for scale, pred in enumerate(preds):
        # 获取当前尺度的锚框
        anchor_set = anchors[scale]

        # 转换预测形状 [batch, grid, grid, anchors, 5 + num_classes]
        output = pred.numpy()
        grid_size = output.shape[1]
        output = output.reshape((output.shape[0], grid_size, grid_size, 3, 5 + num_classes))

        # 遍历每个批次（这里假设batch=1）
        for b in range(output.shape[0]):
            # 遍历每个网格
            for i in range(grid_size):
                for j in range(grid_size):
                    # 遍历每个锚框
                    for a in range(3):
                        # 提取预测值
                        box = output[b, i, j, a]
                        x, y, w, h, conf = box[0], box[1], box[2], box[3], box[4]
                        class_prob = box[5:]

                        # 计算绝对坐标
                        x = (j + sigmoid(x)) / grid_size  # 中心点x坐标
                        y = (i + sigmoid(y)) / grid_size  # 中心点y坐标
                        w = np.exp(w) * anchor_set[a][0] / 416  # 宽度
                        h = np.exp(h) * anchor_set[a][1] / 416  # 高度

                        # 转换到图像坐标系
                        xmin = int((x - w / 2) * 416)
                        ymin = int((y - h / 2) * 416)
                        xmax = int((xmin + w * 416))
                        ymax = int((ymin + h * 416))

                        # 应用置信度阈值
                        conf_score = sigmoid(conf)
                        if conf_score < conf_threshold:
                            continue

                        # 获取类别
                        class_id = np.argmax(class_prob)
                        class_score = class_prob[class_id]

                        # 保存结果
                        boxes.append([xmin, ymin, xmax, ymax])
                        scores.append(float(conf_score * class_score))
                        class_ids.append(class_id)

    # 应用非极大值抑制
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=50,
            iou_threshold=iou_threshold
        ).numpy()

        return boxes[indices], scores[indices], class_ids[indices]
    return [], [], []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualize_detections(image, boxes, scores, class_ids, class_names):
    """可视化检测结果"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., 0], cmap='gray')
    ax = plt.gca()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # 绘制边界框
        rect = Rectangle((x1, y1), width, height,
                         fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

        # 添加标签
        label = f"{class_names[class_id]}: {score:.2f}"
        plt.text(x1, y1 - 5, label, color='red',
                 fontsize=10, fontweight='bold')

    plt.axis('off')
    plt.show()

