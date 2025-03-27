import numpy as np
from keras.applications.convnext import decode_predictions
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
from src.yolo_model import YOLOv4, YOLOv4Loss


def decode_predictions(preds, anchors, num_classes, conf_threshold=0.5, iou_threshold=0.4):
    """将模型输出解码为边界框（带数值稳定性处理）"""
    boxes = []
    scores = []
    class_ids = []
    image_size = 416  # 输入图像尺寸

    for scale, pred in enumerate(preds):
        anchor_set = anchors[scale]

        # 转换预测形状 [batch, grid, grid, anchors, 5 + num_classes]
        output = pred
        grid_size = output.shape[1]
        output = output.reshape((output.shape[0], grid_size, grid_size, 3, 5 + num_classes))

        for b in range(output.shape[0]):
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(3):
                        box = output[b, i, j, a]
                        x, y, w, h, conf = box[0], box[1], box[2], box[3], box[4]
                        class_prob = box[5:]

                        # 数值稳定性处理
                        # 1. 限制坐标偏移量在合理范围
                        x = np.clip(x, -50, 50)
                        y = np.clip(y, -50, 50)

                        # 2. 限制宽高预测值防止指数爆炸
                        w = np.clip(w, -20, 20)
                        h = np.clip(h, -20, 20)

                        # 计算坐标（使用稳定的sigmoid）
                        x_shift = 1 / (1 + np.exp(-x))  # sigmoid
                        y_shift = 1 / (1 + np.exp(-y))
                        x_center = (j + x_shift) / grid_size
                        y_center = (i + y_shift) / grid_size

                        # 计算宽高（带安全保护）
                        w_scaled = np.exp(w) * anchor_set[a][0] / image_size
                        h_scaled = np.exp(h) * anchor_set[a][1] / image_size

                        # 坐标转换（带边界保护）
                        xmin = int(np.clip((x_center - w_scaled / 2) * image_size, 0, image_size - 1))
                        ymin = int(np.clip((y_center - h_scaled / 2) * image_size, 0, image_size - 1))
                        xmax = int(np.clip((x_center + w_scaled / 2) * image_size, 0, image_size - 1))
                        ymax = int(np.clip((y_center + h_scaled / 2) * image_size, 0, image_size - 1))

                        # 处理置信度
                        conf_score = 1 / (1 + np.exp(-conf))
                        if conf_score < conf_threshold:
                            continue

                        # 类别处理
                        class_id = np.argmax(class_prob)
                        class_score = class_prob[class_id]
                        final_score = conf_score * class_score

                        # 保存有效预测
                        if (xmax - xmin) > 1 and (ymax - ymin) > 1:  # 过滤无效框
                            boxes.append([xmin, ymin, xmax, ymax])
                            scores.append(float(final_score))
                            class_ids.append(class_id)

    # 非极大值抑制（带空值保护）
    if len(boxes) == 0:
        return [], [], []

    boxes_array = np.array(boxes)
    scores_array = np.array(scores)

    # 确保数据类型正确
    boxes_tensor = tf.convert_to_tensor(boxes_array, dtype=tf.float32)
    scores_tensor = tf.convert_to_tensor(scores_array, dtype=tf.float32)

    selected_indices = tf.image.non_max_suppression(
        boxes_tensor,
        scores_tensor,
        max_output_size=50,
        iou_threshold=iou_threshold
    ).numpy()

    return boxes_array[selected_indices], scores_array[selected_indices], np.array(class_ids)[selected_indices]


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




if __name__ == '__main__':
    # 测试代码修正
    ANCHORS = [
        [(12,16), (19,36), (40,28)],    # 小尺度
        [(36,75), (76,55), (72,146)],   # 中尺度
        [(142,110), (192,243), (459,401)] # 大尺度
    ]

    def test_data():
        batch_size = 5
        return (
            np.random.rand(batch_size, 416, 416, 1).astype(np.float32),
            [
                # 顺序调整为小->中->大，与模型输出顺序严格对应
                np.random.rand(batch_size, 52, 52, 3, 6),  # 小尺度 (52x52)
                np.random.rand(batch_size, 26, 26, 3, 6),  # 中尺度 (26x26)
                np.random.rand(batch_size, 13, 13, 3, 6)   # 大尺度 (13x13)
            ]
        )

    # 初始化模型
    model = YOLOv4(
        num_classes=1,
        anchors=ANCHORS,
        input_size=(416,416,1)
    )

    # # 模型编译部分修正
    # model.compile(
    #     optimizer='adam',
    #     # 确保input_size参数格式与模型定义一致
    #     loss=YOLOv4Loss(anchors=ANCHORS, num_classes=1, input_size=(416, 416))
    # )

    # 修改模型编译部分(梯度裁切）
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),  # 添加梯度裁剪
        loss=YOLOv4Loss(anchors=ANCHORS, num_classes=1, input_size=416)
    )

    # 执行测试
    test_img, test_label = test_data()
    output = model(test_img)
    print("输出尺寸验证：")
    print(f"小尺度: {output[0].shape} (预期: (5,52,52,18))")
    print(f"中尺度: {output[1].shape} (预期: (5,26,26,18))")
    print(f"大尺度: {output[2].shape} (预期: (5,13,13,18))")

    # 训练测试
    loss = model.train_on_batch(test_img, test_label)
    print(f"训练损失: {loss}")

    # 训练测试
    history = model.fit(test_img, test_label, epochs=100, batch_size=5)


    def create_test_image():
        image = np.zeros((416, 416, 1))
        # 添加矩形目标
        image[100:200, 150:250, 0] = 1.0  # 简单矩形
        return np.expand_dims(image, axis=0)



    # 定义类别（根据你的数据集）
    CLASS_NAMES = ["object"]

    # 执行检测
    test_image = create_test_image()
    preds = model.predict(test_image)

    # 解码预测结果（修正参数顺序）
    boxes, scores, class_ids = decode_predictions(
        preds,
        anchors=ANCHORS,
        num_classes=1,  # 正确传递参数位置
        conf_threshold=0.3
    )

    # 可视化结果
    print(f"检测到 {len(boxes)} 个目标")
    visualize_detections(test_image, boxes, scores, class_ids, CLASS_NAMES)