
import numpy as np
from keras.applications.convnext import decode_predictions
from src.draw import visualize_detections
from src.yolo_model import YOLOv4, YOLOv4Loss

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

# 模型编译部分修正
model.compile(
    optimizer='adam',
    # 确保input_size参数格式与模型定义一致
    loss=YOLOv4Loss(anchors=ANCHORS, num_classes=1, input_size=(416, 416))
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
history = model.fit(test_img, test_label, epochs=10, batch_size=5)



# 使用示例 ---------------------------------------------------
# 生成模拟测试图像（带目标的灰度图像）
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

# 解码预测结果
boxes, scores, class_ids = decode_predictions(
    preds,
    anchors=ANCHORS,
    num_classes=1,
    conf_threshold=0.3  # 可调整的置信度阈值
)

# 可视化结果
print(f"检测到 {len(boxes)} 个目标")
visualize_detections(test_image, boxes, scores, class_ids, CLASS_NAMES)