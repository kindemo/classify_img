import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
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
    x, y, z = voxel_coord  # 正确顺序为(x,y,z)
    return [
        origin[0] + x * spacing[0],
        origin[1] + y * spacing[1],
        origin[2] + z * spacing[2]
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

    import pandas as pd
    import os

    def _load_annotations(self, mhd_path):
        """加载对应CT的真实结节标注"""
        # 示例标注文件路径（根据实际情况调整）
        annotation_file = config.annotation_csv

        if not os.path.exists(annotation_file):
            return []

        # 提取seriesuid
        seriesuid = os.path.basename(mhd_path).split('.mhd')[0]

        # 读取CSV文件
        df = pd.read_csv(annotation_file)
        nodules = df[df['seriesuid'] == seriesuid]

        true_annotations = []
        for _, row in nodules.iterrows():
            true_annotations.append({
                'x': row['coordX'],
                'y': row['coordY'],
                'z': row['coordZ'],
                'diameter': row['diameter_mm']
            })
        return true_annotations

    def _get_annotated_z_levels(self, annotations, origin, spacing, total_slices):
        """从标注中解析需要处理的z层"""
        target_zs = []
        for ann in annotations:
            # 世界坐标转体素坐标
            voxel_z = (ann['z'] - origin[2]) / spacing[2]
            z_index = int(round(voxel_z))

            # 有效性检查
            if 0 <= z_index < total_slices:
                target_zs.append(z_index)

        # 去重并排序
        return sorted(list(set(target_zs)))

    def detect_ct(self, mhd_path):
        """完整检测流程"""
        # 1. 加载CT数据
        ct_array, origin, spacing = self._load_ct(mhd_path)

        # 2. 加载真实结节标注
        true_annotations = self._load_annotations(mhd_path)

        # # 3. 遍历所有轴向切片
        # all_results = []
        # for z in range(ct_array.shape[0]):
        #     slice_data = ct_array[z]
        #
        #     # 3. 滑动窗口检测
        #     detections = self._detect_slice(slice_data, z, spacing)
        #
        #     # 4. 合并当前切片结果
        #     all_results.extend(detections)

        # 3. 计算需要处理的z层
        target_zs = self._get_annotated_z_levels(true_annotations, origin, spacing, ct_array.shape[0])
        if not target_zs:
            print("未找到标注层，跳过检测")
            return []

        # 4. 仅遍历标注层切片
        all_results = []
        for z in target_zs:
            slice_data = ct_array[z]
            detections = self._detect_slice(slice_data, z, spacing, origin)
            all_results.extend(detections)

        # 5. 全局NMS
        final_results = self._nms(all_results)

        # 在NMS后强制将框调整为正方形
        for box in final_results:
            max_size = max(box['width'], box['height'])
            box['width'] = max_size
            box['height'] = max_size

        # 6. 可视化（传递真实标注和坐标信息）
        for i in range(len(target_zs)):
            self._visualize(ct_array, final_results, true_annotations, origin, spacing,
                            target_z=target_zs[i])

        return final_results

    def _detect_slice(self, slice_data, z_index, spacing, origin, window_size=config.input_size):
        """处理单个轴向切片"""
        detections = []
        stride = window_size // 2
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
                boxes = self._decode_predictions(preds, (z_index, y, x), spacing, origin)

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

    def _decode_predictions(self, preds, offset, spacing, origin):
        """修正后的预测解码逻辑"""
        # 按尺度顺序处理输出：large(13x13), medium(26x26), small(52x52)
        output_keys = ['large', 'medium', 'small']

        boxes = []
        z_index, y_start, x_start = offset  # 当前窗口的起始坐标（体素单位）
        num_classes = len(self.config['class_names'])
        input_size = self.config['input_size']  # 模型输入尺寸，如512
        window_size = 256  # 滑动窗口的原始尺寸

        # 计算每个网格单元的实际像素大小
        scale_ratio = window_size / input_size

        for scale_idx, output_key in enumerate(output_keys):
            if output_key not in preds:
                continue

            pred = preds[output_key][0]  # 取第一个batch
            grid_size = pred.shape[1]  # 特征图尺寸（如13,26,52）
            anchors = self.config['anchors'][scale_idx]

            # 调整预测张量形状 (grid, grid, 3, 5+classes)
            if pred.shape[-1] == 3 * (5 + num_classes):
                pred = pred.reshape((grid_size, grid_size, 3, 5 + num_classes))

            # 计算每个网格单元对应的实际像素数
            cell_size = window_size / grid_size

            # 遍历网格单元和锚框
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(3):
                        # 解析置信度
                        conf = sigmoid(pred[i, j, a, 4])
                        print(f'conf: {conf}')
                        if conf < 0.2:  # 过滤低置信度
                            continue

                        # 关键修改点1：正确计算中心坐标
                        tx = sigmoid(pred[i, j, a, 0])
                        ty = sigmoid(pred[i, j, a, 1])

                        # 计算相对于窗口的坐标（像素单位）
                        x_center_window = (j + tx) * cell_size
                        y_center_window = (i + ty) * cell_size

                        # 转换为整个CT切片的坐标（加上窗口起始位置）
                        x_center = x_start + x_center_window
                        y_center = y_start + y_center_window

                        # 关键修改点2：正确计算宽高（毫米单位）
                        bw = np.exp(pred[i, j, a, 2]) * anchors[a][0] / input_size
                        bh = np.exp(pred[i, j, a, 3]) * anchors[a][1] / input_size

                        width = bw * window_size * spacing[0]  # 转换为毫米
                        height = bh * window_size * spacing[1]

                        # 转换为世界坐标（毫米）
                        world_coord = voxel_to_world(
                            (x_center, y_center, z_index),
                            origin=origin,
                            spacing=spacing
                        )

                        # 调试输出
                        print(
                            f"预测框: conf={conf:.2f}, 坐标(z,y,x)=({z_index}, {y_center}, {x_center}), "
                            f"尺寸={width:.1f}x{height:.1f} mm, 世界坐标: {world_coord}"
                        )

                        boxes.append({
                            'z': world_coord[2],  # 世界坐标Z (mm)
                            'y': world_coord[1],  # Y (mm)
                            'x': world_coord[0],  # X (mm)
                            'width': width,
                            'height': height,
                            'confidence': conf
                        })
        return boxes

    def _nms(self, detections, iou_threshold=0.5):
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

    def _visualize(self, ct_array, results, true_annotations, origin, spacing, target_z=91):
        """修正后的可视化方法"""
        if target_z >= ct_array.shape[0]:
            print(f"警告：目标层{target_z}超出CT范围")
            return


        plt.figure(figsize=(16, 12))
        ax = plt.gca()

        # 逆归一化获取原始HU值
        slice_data = ct_array[target_z] * 1400 - 1000
        plt.imshow(slice_data.T, cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"Axial Slice Z={target_z}\nGreen: Predicted | Red: Ground Truth",
                     fontsize=14, color='white', pad=20)
        ax.axis('off')

        # 处理真实标注
        true_boxes = []
        for ann in true_annotations:
            voxel_z = (ann['z'] - origin[2]) / spacing[2]
            z_index = int(round(voxel_z))


            if z_index == target_z:
                # 转换为像素坐标
                voxel_x = (ann['x'] - origin[0]) / spacing[0]
                voxel_y = (ann['y'] - origin[1]) / spacing[1]
                diameter_pixels = ann['diameter'] / spacing[0]  # 假设各向同性

                # 绘制红色框
                rect = Rectangle(
                    (voxel_x - diameter_pixels / 2, voxel_y - diameter_pixels / 2),
                    diameter_pixels, diameter_pixels,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                plt.text(voxel_x + 5, voxel_y + 15, 'True',
                         color='red', fontsize=12,
                         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            print(f"当前可视化层: {target_z}, 预测框层: {z_index}")

        # 处理预测结果
        for box in results:
            # 关键修改点3：世界坐标转像素坐标
            voxel_z = (box['z'] - origin[2]) / spacing[2]
            if abs(voxel_z - target_z) > 0.5:  # 检查是否在当前层
                continue
            # 转换到像素坐标
            voxel_x = (box['x'] - origin[0]) / spacing[0]
            voxel_y = (box['y'] - origin[1]) / spacing[1]
            size_pixels = box['width'] / spacing[0]  # 转换为像素尺寸

            # 绘制绿色框
            # print(f"预测框显示坐标: x={voxel_x}, y={voxel_y}, size={size_pixels}")
            rect = Rectangle(
                (voxel_x - size_pixels / 2, voxel_y - size_pixels / 2),
                size_pixels, size_pixels,
                linewidth=2, edgecolor='#00FF00', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(voxel_x + 5, voxel_y + 15, f"{box['confidence']:.2f}",
                     color='#00FF00', fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        plt.tight_layout()
        plt.show()


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    configs = {
        'input_size': config.input_size,
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
        "D:\BaiduNetdiskDownload\LUNA16\subset0\\1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd"
    )


