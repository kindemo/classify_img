import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from src.config import config
from src.yolo_model import YoloLoss


# -------------------- 缺失函数补全 --------------------
def sliding_window_3d(volume, window_size=256, stride=128):
    """生成三维滑动窗口"""
    for z in range(0, volume.shape[0] - window_size + 1, stride):
        for y in range(0, volume.shape[1] - window_size + 1, stride):
            for x in range(0, volume.shape[2] - window_size + 1, stride):
                yield (
                    volume[z:z + window_size, y:y + window_size, x:x + window_size],
                    (z, y, x)
                )


def voxel_to_world(voxel_coord, origin, spacing):
    """体素坐标转世界坐标 (x,y,z顺序适配CT标准)"""
    return [
        origin[0] + voxel_coord[0] * spacing[0],
        origin[1] + voxel_coord[1] * spacing[1],
        origin[2] + voxel_coord[2] * spacing[2]
    ]


def sigmoid(x):
    """数值稳定的sigmoid实现"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x))
                    )





class CTNoduleDetector:
    def __init__(self, model_path, config):
        # 确保导入所有自定义层
        from src.yolo_model import YOLOv4, CSPDarknet53, PANet, YOLOHead

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'YoloLoss': YoloLoss,
                'CSPDarknet53': CSPDarknet53,
                'PANet': PANet,
                'YOLOHead': YOLOHead,
            },
            compile=False  # 暂时禁用编译以绕过损失函数问题
        )
        # 显式重新编译模型
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=YoloLoss(  # 使用正确的参数初始化
                anchors=config['anchors'],
                num_classes=len(config['class_names'])
            )
        )
        self.config = config
        self.colors = {'nodule': '#00FF00'}

    def _load_ct(self, mhd_path):
        """专业CT数据加载"""
        ct_image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(ct_image)  # (Z,Y,X)

        # 专业肺窗预处理
        array = np.clip(array, -1000, 400).astype(np.float32)
        array = (array + 1000) / 1400  # 归一化到[0,1]

        # 在_load_ct方法中添加验证
        print("CT数据维度:", array.shape)
        print("像素值范围:", np.min(array), np.max(array))

        return array, ct_image.GetOrigin(), ct_image.GetSpacing()

    def detect_ct(self, mhd_path):
        """完整检测流程"""
        # 1. 加载CT数据
        ct_array, origin, spacing = self._load_ct(mhd_path)

        # 2. 遍历所有轴向切片
        all_results = []
        for z in range(ct_array.shape[0]):
            slice_data = ct_array[z]

            # 3. 滑动窗口检测
            detections = self._detect_slice(slice_data, z, spacing)

            # 4. 合并当前切片结果
            all_results.extend(detections)

        # 5. 全局NMS
        final_results = self._nms(all_results)

        # 6. 可视化关键切片
        self._visualize(ct_array, final_results, target_z=91)

        return final_results

    def _detect_slice(self, slice_data, z_index, spacing, window_size=416, stride=256):
        """处理单个轴向切片"""
        detections = []
        h, w = slice_data.shape

        # 滑动窗口处理
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # 提取窗口区域
                window = slice_data[y:y + window_size, x:x + window_size]

                # 预处理
                processed = self._preprocess(window)

                # 模型预测
                preds = self.model.predict(processed)

                # 解码预测结果
                boxes = self._decode_predictions(preds, (z_index, y, x), spacing)

                detections.extend(boxes)

        # 在_detect_slice方法中，预测后添加：
        # print("预测输出类型:", type(preds))
        # print("输出键:", preds.keys())
        # for key in preds:
        #     print(f"{key} 形状:", preds[key].shape)

        return detections

    def _preprocess(self, window):
        """预处理适配示例图像CT值范围"""
        # 标准化到[-1000,400] HU
        window = np.clip(window, -1000, 400)
        # 转换为0-1范围
        window = (window + 1000) / 1400
        # 调整尺寸和维度
        resized = cv2.resize(window, (self.config['input_size'], self.config['input_size']))
        return np.expand_dims(resized, axis=(0, -1))  # 形状: (1,416,416,1)

    def _decode_predictions(self, preds, offset, spacing):

        """处理字典类型的模型输出"""
        # 按尺度顺序处理输出：large(13x13), medium(26x26), small(52x52)
        output_keys = ['large', 'medium', 'small']

        boxes = []
        z, y_start, x_start = offset
        num_classes = len(self.config['class_names'])

        for scale_idx, output_key in enumerate(output_keys):
            if output_key not in preds:
                continue

            # 获取当前尺度参数
            pred = preds[output_key][0]  # 取第一个batch
            grid_size = pred.shape[1]  # 特征图实际尺寸
            anchors = self.config['anchors'][scale_idx]
            print(f"grid_size: {grid_size}, pred_shape: {pred.shape}")

            # 调整预测张量形状 (grid, grid, 3, 5+classes)
            if pred.shape[-1] == 3 * (5 + num_classes):
                pred = pred.reshape((grid_size, grid_size, 3, 5 + num_classes))

            # 遍历网格单元和锚框
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(3):
                        # 解析置信度
                        conf = sigmoid(pred[i, j, a, 4])
                        print(f"conf: {conf}")
                        if conf < 0.5:
                            continue

                        # 解析相对坐标
                        tx = sigmoid(pred[i, j, a, 0])
                        ty = sigmoid(pred[i, j, a, 1])
                        bx = (tx + j) / grid_size  # 网格内偏移
                        by = (ty + i) / grid_size

                        # 计算绝对尺寸
                        bw = np.exp(pred[i, j, a, 2]) * anchors[a][0]
                        bh = np.exp(pred[i, j, a, 3]) * anchors[a][1]

                        # 转换为实际坐标（输入图像尺度）
                        # 坐标转换时加入缩放
                        # 计算窗口实际尺寸与模型输入的比例
                        window_size = 256  # 滑动窗口原始尺寸
                        scale_ratio = window_size / self.config['input_size']
                        x_center_in_window = bx * self.config['input_size']  # 模型输入尺度坐标
                        x_center = x_start + x_center_in_window * scale_ratio  # 缩放回原始窗口

                        # x_center = bx * self.config['input_size'] + x_start
                        y_center = by * self.config['input_size'] + y_start
                        width = bw * self.config['input_size'] * scale_ratio * spacing[2]  # 最终物理尺寸
                        height = bh * self.config['input_size'] * scale_ratio * spacing[1]

                        # 转换到世界坐标系（毫米）
                        world_coord = voxel_to_world(
                            (z, y_center, x_center),  # 注意CT坐标顺序(Z,Y,X)
                            origin=(0, 0, 0),  # 根据实际原点调整
                            spacing=spacing
                        )

                        boxes.append({
                            'z': z,
                            'y': y_center,
                            'x': x_center,
                            'width': width * spacing[2],  # X方向间距
                            'height': height * spacing[1],  # Y方向间距
                            'confidence': conf
                        })
        return boxes

    def _nms(self, detections, iou_threshold=0.4):
        """二维非极大值抑制"""
        # 按z轴分组处理
        z_groups = {}
        for det in detections:
            z = det['z']
            z_groups.setdefault(z, []).append(det)

        # 每个切片单独处理
        keep = []
        for z, boxes in z_groups.items():
            # 转换为numpy数组
            boxes_array = np.array([[
                b['x'] - b['width'] / 2,
                b['y'] - b['height'] / 2,
                b['x'] + b['width'] / 2,
                b['y'] + b['height'] / 2,
                b['confidence']
            ] for b in boxes])

            # 执行NMS
            indices = tf.image.non_max_suppression(
                boxes_array[:, :4],
                boxes_array[:, 4],
                max_output_size=50,
                iou_threshold=iou_threshold
            ).numpy()

            # 保留结果
            keep.extend([boxes[i] for i in indices])

        return keep

    def _visualize(self, ct_array, results, target_z=91):
        plt.figure(figsize=(16, 12))
        ax = plt.gca()

        # 逆归一化获取原始HU值
        slice_data = ct_array[target_z] * 1400 - 1000
        print("slice_data范围:", np.min(slice_data), np.max(slice_data))

        plt.imshow(slice_data.T, cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"Axial Slice Z={target_z}", fontsize=14, color='white', pad=20)
        ax.axis('off')

        # 筛选当前切片的检测结果
        valid_boxes = [b for b in results if b['z'] == target_z]
        print(f"在切片Z={target_z}发现{len(valid_boxes)}个结节")

        for box in valid_boxes:
            # 转换到显示坐标系 (X <-> Y交换)
            x_center = box['x']
            y_center = box['y']
            width = box['width']
            height = box['height']

            x1 = y_center - height / 2
            y1 = x_center - width / 2
            display_width = height
            display_height = width

            rect = Rectangle((x1, y1), display_width, display_height,
                             linewidth=2, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect)

            # 添加置信度文本
            text = f"{min(box['confidence'], 0.99):.2f}"
            plt.text(x1 + 5, y1 + 15, text, color='#00FF00', fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        plt.tight_layout()
        plt.show()


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    configs = {
        'input_size': 416,
        'anchors': config.MODEL["anchors"],  # 与训练配置一致
        'class_names': ['nodule']
    }

    """
        :param model_path: 训练好的YOLOv4模型路径
        :param config: 配置字典，包含：
            - input_size: 模型输入尺寸(默认416)
            - anchors: 各尺度锚框配置
            - class_names: 类别名称列表
    """

    detector = CTNoduleDetector(
        model_path="D:/PycharmProjects/classify_img/model",
        config=configs
    )

    results = detector.detect_ct(
        "D:\BaiduNetdiskDownload\LUNA16\subset0\\1.3.6.1.4.1.14519.5.2.1.6279.6001.238522526736091851696274044574.mhd"
    )


