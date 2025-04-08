import anchors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import SimpleITK as sitk
from scipy import ndimage

from src.config import config
from src.preprocessor import LunaYoloPreprocessor
from src.yolo_model import YOLOv4
from src.yolo_model import YoloLoss  # 导入自定义损失函数
from src.yolo_model import CSPDarknet53, PANet, YOLOHead  # 导入所有自定义层


def decode_predictions(preds_dict, anchors, num_classes, conf_threshold=0.5, iou_threshold=0.4):
    """改进的解码函数，处理字典格式的预测结果"""
    boxes = []
    scores = []
    class_ids = []

    # 定义尺度处理顺序（必须与模型输出顺序一致）
    scale_order = ['large', 'medium','small']  # 根据YOLOv4实际输出顺序调整

    for scale_name in scale_order:
        pred = preds_dict[scale_name]
        anchor_set = anchors[scale_order.index(scale_name)]  # 根据顺序获取对应锚框

        # 转换预测形状 [batch, grid, grid, anchors, 5 + num_classes]
        output = pred
        grid_size = output.shape[1]
        output = output.reshape((-1, grid_size, grid_size, 3, 5 + num_classes))

        # 生成网格坐标
        grid_y, grid_x = np.mgrid[0:grid_size, 0:grid_size]
        grid = np.stack((grid_x, grid_y), axis=-1)[..., None, :]  # (g, g, 1, 2)

        # 解码预测值
        xy = (sigmoid(output[..., 0:2]) + grid) / grid_size  # 中心坐标 (0~1范围)
        wh = np.exp(output[..., 2:4]) * anchor_set / 416  # 相对输入尺寸的宽高

        # 计算绝对坐标
        x_center = xy[..., 0] * 416  # 乘以输入尺寸
        y_center = xy[..., 1] * 416
        box_w = wh[..., 0]
        box_h = wh[..., 1]

        # 转换到图像坐标系
        xmin = (x_center - box_w / 2).astype(int)
        ymin = (y_center - box_h / 2).astype(int)
        xmax = (x_center + box_w / 2).astype(int)
        ymax = (y_center + box_h / 2).astype(int)

        # 置信度和类别处理
        conf = sigmoid(output[..., 4])
        class_prob = sigmoid(output[..., 5:5 + num_classes])

        # 遍历所有预测
        for b in range(output.shape[0]):
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(3):  # 每个位置3个锚框
                        conf_score = conf[b, i, j, a]
                        if conf_score < conf_threshold:
                            continue

                        # 获取最大概率类别
                        cls_id = np.argmax(class_prob[b, i, j, a])
                        cls_score = np.max(class_prob[b, i, j, a])

                        # 计算综合得分
                        total_score = conf_score * cls_score

                        # 保存结果
                        boxes.append([
                            xmin[b, i, j, a],
                            ymin[b, i, j, a],
                            xmax[b, i, j, a],
                            ymax[b, i, j, a]
                        ])
                        scores.append(float(total_score))
                        class_ids.append(cls_id)

    # 应用非极大值抑制
    if len(boxes) > 0:
        indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=50,
            iou_threshold=iou_threshold
        ).numpy()

        return (
            np.array(boxes)[indices],
            np.array(scores)[indices],
            np.array(class_ids)[indices]
        )
    return [], [], []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import cv2


def visualize_predictions(mhd_path, coord_world, model, anchors, num_classes, input_size=416):
    """直接处理原始CT数据并可视化预测结果"""
    try:
        # 1. 读取CT扫描数据

        ct_image = sitk.ReadImage(mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_image)  # 获取numpy数组 (z, y, x)

        # 读取CT数据后添加预处理
        ct_array = np.clip(ct_array, a_min=-1000, a_max=400)  # 肺窗范围
        ct_array = (ct_array + 1000) / 1400  # 归一化到0~1

        origin = ct_image.GetOrigin()  # 原始坐标原点 (x,y,z)
        spacing = ct_image.GetSpacing()  # 体素间距 (x,y,z)

        # 2. 坐标转换（世界坐标 -> 体素坐标）
        coord_voxel = LunaYoloPreprocessor._world_to_voxel(coord_world, origin, spacing)
        print(f"世界坐标 {coord_world} -> 体素坐标 {coord_voxel}")

        # 3. 提取立方体


        cube_size = config.cube_size
        cube = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)

        # 计算实际提取范围
        z_start = max(coord_voxel[0] - cube_size // 2, 0)
        z_end = min(z_start + cube_size, ct_array.shape[0])
        y_start = max(coord_voxel[1] - cube_size // 2, 0)
        y_end = min(y_start + cube_size, ct_array.shape[1])
        x_start = max(coord_voxel[2] - cube_size // 2, 0)
        x_end = min(x_start + cube_size, ct_array.shape[2])

        # 修正立方体提取逻辑
        # 计算各维度实际提取长度
        z_length = z_end - z_start
        y_length = y_end - y_start
        x_length = x_end - x_start

        # 动态计算填充区域
        cube_z_slice = slice(0, z_length)
        cube_y_slice = slice(0, y_length)
        cube_x_slice = slice(0, x_length)

        # 确保有有效数据才进行填充
        if z_length > 0 and y_length > 0 and x_length > 0:
            cube[cube_z_slice, cube_y_slice, cube_x_slice] = \
                ct_array[z_start:z_end, y_start:y_end, x_start:x_start + x_length]
        else:
            raise  ValueError("立方体提取区域无效，请检查坐标是否在有效范围内")


        # 4. 多尺度处理（以中等尺度为例）
        scale = 1  # 使用中等尺度（对应26x26网格）
        target_size = config.grid_sizes[scale]
        zoom_factor = [
            target_size / cube.shape[0],
            target_size / cube.shape[1],
            target_size / cube.shape[2]
        ]
        scaled_cube = ndimage.zoom(cube, zoom_factor, order=1)

        # 5. 提取中心切片
        z_center = scaled_cube.shape[0] // 2
        slice_img = scaled_cube[z_center]

        # 6. 预处理（与训练一致）
        processed_img = cv2.resize(slice_img, (input_size, input_size))
        processed_img = np.expand_dims(processed_img, axis=-1)  # 添加通道维度
        processed_img = processed_img.astype(np.float32) / 255.0  # 归一化

        # 7. 模型预测
        input_array = tf.expand_dims(processed_img, axis=0)
        preds_dict = model.predict(input_array)
        print("预测结果类型:", type(preds_dict))  # 应该显示 <class 'list'>
        print("每个输出的类型:", [type(p) for p in preds_dict])  # 应该都是 <class 'tensorflow.python.framework.ops.EagerTensor'>

        boxes, scores, class_ids = decode_predictions(
            preds_dict,  # 传入字典
            anchors=anchors,
            num_classes=num_classes,
            conf_threshold=0.5,
            iou_threshold=0.4
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(processed_img[..., 0], cmap='gray')
        current_axis = plt.gca()

        for box, score, cls_id in zip(boxes, scores, class_ids):
            xmin, ymin, xmax, ymax = box
            current_axis.add_patch(Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='lime', facecolor='none'))

            label = f"{1}: {score:.2f}"
            current_axis.text(
                xmin, ymin - 5, label,
                color='white', fontsize=8,
                bbox=dict(facecolor='lime', alpha=0.7)
            )

        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"预测失败: {str(e)}")
        raise



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



# 使用示例（需要指定结节世界坐标）
visualize_predictions(
    mhd_path="D:\BaiduNetdiskDownload\LUNA16\subset0\\1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd",
    coord_world=(145.967465, -161.1976342, -312.0713474),  # 示例坐标，需替换为真实结节坐标
    model=model,
    anchors=config.MODEL["anchors"],
    num_classes=config.num_classes
)