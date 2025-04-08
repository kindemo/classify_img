import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

from src.config import config
from src.yolo_model import YOLOv4
from src.yolo_model import YoloLoss  # 导入自定义损失函数
from src.yolo_model import CSPDarknet53, PANet, YOLOHead  # 导入所有自定义层


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


import cv2


def visualize_predictions(image_path, model, anchors, num_classes, input_size=416):
    # 1. 预处理图像
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image, (input_size, input_size))
    input_array = tf.expand_dims(input_image / 255.0, axis=0)  # 归一化并添加batch维度

    # 2. 模型预测
    preds = model.predict(input_array)

    # 3. 解码预测结果（修正版）
    boxes, scores, class_ids = decode_predictions(
        preds, anchors, num_classes,
        conf_threshold=0.6,  # 调高置信度阈值
        iou_threshold=0.5  # 调整NMS阈值
    )

    # 4. 可视化
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    current_axis = plt.gca()

    # 绘制检测框
    for box, score, cls_id in zip(boxes, scores, class_ids):
        # 转换坐标到原始图像尺寸
        xmin = int(box[0] * orig_w / input_size)
        ymin = int(box[1] * orig_h / input_size)
        xmax = int(box[2] * orig_w / input_size)
        ymax = int(box[3] * orig_w / input_size)

        # 绘制矩形框
        current_axis.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='lime', facecolor='none'))

        # 添加标签
        label = f"{config.class_names[cls_id]}: {score:.2f}"
        current_axis.text(
            xmin, ymin - 10, label,
            color='white', fontsize=10,
            bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none')
        )

    plt.axis('off')
    plt.show()


# 使用示例

model = tf.keras.models.load_model(
    "D:\\PycharmProjects\\classify_img\\model",
    custom_objects={
        'YoloLoss': YoloLoss,  # 关键：注册损失函数
        'CSPDarknet53': CSPDarknet53,
        'PANet': PANet,
        'YOLOHead': YOLOHead
    }
)



visualize_predictions(
    "test_image.jpg",
    model,
    anchors=config.MODEL["anchors"],
    num_classes=config.num_classes
)

