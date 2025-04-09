import glob
import os
import math
from pathlib import Path
from src.config import config
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import cv2
import tensorflow as tf
from PIL.ImageOps import scale
from keras.callbacks import ModelCheckpoint
from scipy import ndimage

from src.YoloLabelGenerator import YOLOLabelGenerator
from src.config import config
from src.yolo_model import CSPDarknet53, PANet, YOLOHead


def validate_dataset(config):
    """完整的数据集验证"""
    image_files = glob.glob(f"{config.image_dir}/*.png")
    print(f"发现{len(image_files)}张图像")

    missing_labels = 0
    corrupted_files = 0

    for img_path in image_files:
        # 转换标签路径
        label_path = Path(img_path.replace("images", "labels")).with_suffix(".txt")

        # 检查标签文件存在性
        if not label_path.exists():
            print(f"缺失标签文件: {label_path}")
            missing_labels += 1
            continue

        # 检查文件可读性
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 验证内容格式
                for line in content.splitlines():
                    if not validate_label_line(line):
                        print(f"无效标签行: {line} 在文件 {label_path}")
                        corrupted_files += 1
        except UnicodeDecodeError:
            print(f"文件编码错误: {label_path}")
            corrupted_files += 1
        except Exception as e:
            print(f"读取文件异常: {label_path}, 错误: {str(e)}")
            corrupted_files += 1

    print(f"验证完成，缺失标签: {missing_labels}，损坏文件: {corrupted_files}")


def validate_label_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        if width <= 0 or height <= 0:
            return False
    except ValueError:
        return False
    return True


def create_dataset(config, batch_size):
    """构建YOLOv4训练数据集，返回(image, (large_label, medium_label, small_label))格式"""

    def _parse_yolo_data(img_path):
        """解析单样本：读取图像+处理标签"""
        # 读取图像
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)  # 灰度图像单通道
        img = tf.image.resize(img, (config.input_size, config.input_size))
        img = tf.cast(img, tf.float32) / 255.0  # 归一化到[0,1]
        img.set_shape([config.input_size, config.input_size, 1])  # 显式设置形状

        # 生成标签路径
        label_path = tf.strings.regex_replace(img_path, "images", "labels")
        label_path = tf.strings.regex_replace(label_path, "\.png$", ".txt")

        # 解析标签内容
        def _parse_label_content(content):
            """将标签文本解析为三个检测层的张量（TensorFlow安全版本）"""
            # 初始化空标签张量
            large_label = tf.zeros((13, 13, 3, 6), dtype=tf.float32)
            medium_label = tf.zeros((26, 26, 3, 6), dtype=tf.float32)
            small_label = tf.zeros((52, 52, 3, 6), dtype=tf.float32)

            # 处理空内容
            content = tf.strings.strip(content)
            is_empty = tf.equal(tf.strings.length(content), 0)

            def process_non_empty():
                lines = tf.strings.split(content, '\n')
                line_count = tf.shape(lines)[0]

                # 使用while_loop替代for循环
                def process_lines(i, large, medium, small):
                    line = tf.strings.strip(lines[i])

                    # 跳过空行
                    return tf.cond(
                        tf.equal(tf.strings.length(line), 0),
                        lambda: (i + 1, large, medium, small),
                        lambda: process_valid_line(i, large, medium, small, line)
                    )

                def process_valid_line(i, large, medium, small, line):
                    parts = tf.strings.split(line)

                    # 验证字段数量
                    return tf.cond(
                        tf.not_equal(tf.shape(parts)[0], 10),
                        lambda: (i + 1, large, medium, small),  # 跳过无效行
                        lambda: parse_and_update(i, large, medium, small, parts)
                    )

                def parse_and_update(i, large, medium, small, parts):
                    # 解析字段
                    scale = tf.strings.to_number(parts[0], tf.int32)
                    grid_x = tf.strings.to_number(parts[1], tf.int32)
                    grid_y = tf.strings.to_number(parts[2], tf.int32)
                    anchor_idx = tf.strings.to_number(parts[3], tf.int32)

                    # 数值范围约束
                    grid_x = tf.clip_by_value(grid_x, 0, 51)  # 最大支持small尺度52x52
                    grid_y = tf.clip_by_value(grid_y, 0, 51)
                    anchor_idx = tf.clip_by_value(anchor_idx, 0, 2)

                    # 构建更新数据
                    update = tf.stack([
                        tf.strings.to_number(parts[4], tf.float32),
                        tf.strings.to_number(parts[5], tf.float32),
                        tf.strings.to_number(parts[6], tf.float32),
                        tf.strings.to_number(parts[7], tf.float32),
                        tf.strings.to_number(parts[8], tf.float32),
                        tf.cast(tf.strings.to_number(parts[9], tf.int32), tf.float32)
                    ])

                    # 选择目标尺度
                    target, grid_size = tf.switch_case(
                        scale,
                        [
                            lambda: (large, 13),
                            lambda: (medium, 26),
                            lambda: (small, 52)
                        ]
                    )

                    # 生成更新索引
                    indices = tf.stack([
                        tf.clip_by_value(grid_y, 0, grid_size - 1),
                        tf.clip_by_value(grid_x, 0, grid_size - 1),
                        tf.clip_by_value(anchor_idx, 0, 2)
                    ])
                    indices = tf.reshape(indices, [1, 3])  # 转换为二维索引

                    # 执行张量更新
                    updated_target = tf.tensor_scatter_nd_update(
                        target,
                        indices,
                        tf.reshape(update, [1, 6])
                    )

                    # 返回更新后的张量
                    return tf.switch_case(
                        scale,
                        [
                            lambda: (i + 1, updated_target, medium, small),
                            lambda: (i + 1, large, updated_target, small),
                            lambda: (i + 1, large, medium, updated_target)
                        ]
                    )

                # 执行循环处理
                final_i, final_large, final_medium, final_small = tf.while_loop(
                    cond=lambda i, *_: i < line_count,
                    body=process_lines,
                    loop_vars=(0, large_label, medium_label, small_label),
                    shape_invariants=(
                        tf.TensorShape([]),
                        tf.TensorShape([13, 13, 3, 6]),
                        tf.TensorShape([26, 26, 3, 6]),
                        tf.TensorShape([52, 52, 3, 6])
                    )
                )

                return final_large, final_medium, final_small

            # 主条件判断
            return tf.cond(
                is_empty,
                lambda: (large_label, medium_label, small_label),
                process_non_empty
            )

        # 读取并处理标签
        label_content = tf.io.read_file(label_path)
        large, medium, small = tf.py_function(
            _parse_label_content,
            [label_content],
            [tf.float32, tf.float32, tf.float32]
        )

        # 显式设置形状（使用tf.ensure_shape确保形状）
        large = tf.ensure_shape(large, (13, 13, 3, 6))
        medium = tf.ensure_shape(medium, (26, 26, 3, 6))
        small = tf.ensure_shape(small, (52, 52, 3, 6))

        # 添加断言验证形状
        tf.debugging.assert_shapes([
            (large, (13, 13, 3, 6)),
            (medium, (26, 26, 3, 6)),
            (small, (52, 52, 3, 6))
        ])

        return img, {
            "large": large,
            "medium": medium,
            "small": small
        }  # 标签改为字典格式

    # 构建数据集管道
    img_files = tf.data.Dataset.list_files(f"{config.image_dir}/*.png", shuffle=True)
    dataset = img_files.map(
        _parse_yolo_data,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 数据集优化
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)




class LunaYoloPreprocessor():
    def __init__(self, config):
        self.config = config
        self._create_dirs()
        self.file_counter = 0
        self.label_generator = YOLOLabelGenerator(config)  # 组合标签生成器

        # 确保路径正确
        self.config.image_dir = str(Path(self.config.processed_dir) / "images")
        self.config.label_dir = str(Path(self.config.processed_dir) / "labels")

    def _create_dirs(self):
        """创建图像和标签目录，并验证可写性"""
        images_dir = Path(self.config.image_dir)
        labels_dir = Path(self.config.label_dir)

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 验证可写性
        test_file = images_dir / "test.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise PermissionError(f"目录不可写: {images_dir}，错误: {str(e)}")

    def _extract_cube(self, img_3d, center_voxel):
        """提取3D立方体（确保z,y,x顺序与CT数据对齐）"""
        size = self.config.cube_size
        z, y, x = center_voxel  # 已修正为z,y,x顺序

        # 计算各维度起止索引
        z_start = max(z - size // 2, 0)
        z_end = min(z_start + size, img_3d.shape[0])
        y_start = max(y - size // 2, 0)
        y_end = min(y_start + size, img_3d.shape[1])
        x_start = max(x - size // 2, 0)
        x_end = min(x_start + size, img_3d.shape[2])

        # 处理边界溢出
        cube = np.zeros((size, size, size), dtype=img_3d.dtype)

        # 计算实际填充区域
        actual_z_slice = slice(z_start, z_end)
        actual_y_slice = slice(y_start, y_end)
        actual_x_slice = slice(x_start, x_end)

        # 填充数据
        cube_z_start = max(0, (size // 2) - (z - z_start))
        cube_z_end = cube_z_start + (z_end - z_start)

        cube[cube_z_start:cube_z_end, :, :] = img_3d[actual_z_slice, actual_y_slice, actual_x_slice]
        return cube

    def _generate_scale_data(self, cube, annot, patient_id, scale):
        """生成不同尺度的图像和标签"""
        target_size = self.config.grid_sizes[scale]
        cube_size = self.config.cube_size

        # 三维缩放
        zoom_factor = [
            target_size / cube.shape[0],
            target_size / cube.shape[1],
            target_size / cube.shape[2]
        ]
        scaled_cube = ndimage.zoom(cube, zoom_factor, order=1)

        # 假设结节占据整个立方体，归一化尺寸为1.0（根据实际需求调整）
        norm_x = 0.5  # 中心坐标
        norm_y = 0.5
        norm_w = 1.0  # 占据整个图像宽度
        norm_h = 1.0  # 占据整个图像高度

        # 生成所有切片
        for z in range(scaled_cube.shape[0]):
            img_name = f"{patient_id}_s{scale}_z{z}.png"
            img_path = os.path.join(self.config.image_dir, img_name)
            cv2.imwrite(img_path, scaled_cube[z])

            # 构造标签信息
            slice_annot = {
                'seriesuid': img_name.replace('.png', ''),
                'x_center': norm_x,
                'y_center': norm_y,
                'width': norm_w,
                'height': norm_h
            }
            # print(f"生成图像: {img_path}, 形状: {scaled_cube[z].shape}")

            # 生成标签
            self.label_generator.create_scale_label(img_path, slice_annot, scale)
            # print(f"生成标签: {img_path.replace('images', 'labels').replace('.png', '.txt')}")

    @staticmethod
    def _world_to_voxel(world_coord, origin, spacing):
        """将世界坐标(x,y,z)转换为体素坐标(z,y,x)索引"""
        voxel_x = (world_coord[0] - origin[0]) / spacing[0]
        voxel_y = (world_coord[1] - origin[1]) / spacing[1]
        voxel_z = (world_coord[2] - origin[2]) / spacing[2]
        return np.round([voxel_z, voxel_y, voxel_x]).astype(int)  # 适配数组(z,y,x)顺序


    def process_nodule(self, ct_scan, annot, patient_id):
        """处理单个结节"""
        world_coord = np.array([annot['coordX'], annot['coordY'], annot['coordZ']])
        voxel_coord = self._world_to_voxel(world_coord, ct_scan['origin'], ct_scan['spacing'])
        print(f"转换后体素坐标: {voxel_coord}, CT数据形状: {ct_scan['data'].shape}")

        if not self._is_valid_coordinate(voxel_coord, ct_scan['data'].shape):
            print(f"无效坐标跳过: {voxel_coord}")
            return

        cube = self._extract_cube(ct_scan['data'], voxel_coord)

        # 多尺度处理
        for scale in range(len(self.config.grid_sizes)):
            self._generate_scale_data(cube, annot, patient_id, scale)

    def _is_valid_coordinate(self, coord, data_shape):
        """验证坐标有效性（data_shape应为z,y,x顺序）"""
        # coord的维度顺序应为z,y,x
        z, y, x = coord
        return (0 <= z < data_shape[0]) and (0 <= y < data_shape[1]) and (0 <= x < data_shape[2])


# 关注所有切片
class FullSlicePreprocessor:
    def __init__(self, config):
        self.config = config
        self.slice_counter = 0
        self.label_generator = YOLOLabelGenerator(config)


        # 创建输出目录
        self.image_dir = Path(config.processed_dir) / "images"
        self.label_dir = Path(config.processed_dir) / "labels"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.origin  = None
        self.spacing = None  # 新增spacing属性

    def process_patient(self, mhd_path, annotations):
        """处理单个患者的全部CT切片"""
        # 读取CT数据
        ct_scan = sitk.ReadImage(mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_scan)  # shape: (num_slices, height, width)

        origin = ct_scan.GetOrigin()
        spacing = ct_scan.GetSpacing()

        self.origin = origin  # 新增origin存储
        self.spacing = spacing

        # 遍历所有切片
        for slice_idx in range(ct_array.shape[0]):
            self.process_slice(
                ct_slice=ct_array[slice_idx],
                slice_z=origin[2] + slice_idx * spacing[2],  # 当前切片的Z轴坐标
                spacing=spacing,
                patient_id=Path(mhd_path).stem,
                annotations=annotations,
                slice_idx=slice_idx
            )

    def process_slice(self, ct_slice, slice_z, spacing, patient_id, annotations, slice_idx):
        """处理单个切片"""
        # 转换CT值为灰度图像
        windowed = self.apply_window(ct_slice, self.config.hu_window)
        png_img = ((windowed - windowed.min()) / (windowed.max() - windowed.min()) * 255).astype(np.uint8)

        # 生成图像文件名
        img_name = f"{patient_id}_slice{slice_idx:04d}.png"
        img_path = self.image_dir / img_name

        # 保存图像
        cv2.imwrite(str(img_path), png_img)

        # 生成对应标签
        self.generate_labels(
            img_path=img_path,
            slice_z=slice_z,
            annotations=annotations,
            img_size=ct_slice.shape,
            spacing=spacing
        )

    def apply_window(self, image, window):
        """应用CT窗宽窗位"""
        min_val = window[0] - window[1] / 2.0
        max_val = window[0] + window[1] / 2.0
        return np.clip(image, min_val, max_val)

    def generate_labels(self, img_path, slice_z, annotations, img_size, spacing):
        """生成YOLO格式标签文件（支持多尺度）"""
        labels = []
        img_w = img_size[1]  # 图像宽度（像素）
        img_h = img_size[0]  # 图像高度（像素）

        # 转换物理坐标系到像素坐标系
        for _, annot in annotations.iterrows():
            # 仅处理当前切片附近的标注
            if abs(annot['coordZ'] - slice_z) > spacing[2] * self.config.slice_thickness:
                continue


            # 坐标转换
            x_center_px = (annot['coordX'] - self.origin[0]) / spacing[0]
            y_center_px = (annot['coordY'] - self.origin[1]) / spacing[1]
            width_px = annot['diameter_mm'] / spacing[0]
            height_px = annot['diameter_mm'] / spacing[1]

            # 归一化处理
            x_center = x_center_px / img_w
            y_center = y_center_px / img_h
            width = width_px / img_w
            height = height_px / img_h

            # 数值验证
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                print(f"坐标超出范围: {x_center}, {y_center}")
                continue
            if width <= 0 or height <= 0:
                print(f"无效尺寸: {width}, {height}")
                continue

            labels.append([
                self.config.class_id,
                x_center,
                y_center,
                width,
                height
            ])


        # 保存标签文件（强制UTF-8编码）
        label_path = self.label_dir / f"{img_path.stem}.txt"
        try:
            if labels:
                # 格式验证
                assert all(len(item) == 5 for item in labels), "标签维度错误"
                np.savetxt(
                    str(label_path),
                    labels,
                    fmt="%d %.6f %.6f %.6f %.6f",
                    header='',
                    comments='',
                    encoding='utf-8'
                )
            else:
                # 创建空文件防止FileNotFoundError
                label_path.write_text("", encoding='utf-8')
        except Exception as e:
            print(f"保存标签失败: {label_path}, 错误: {str(e)}")
            raise


# 只关注极少量切片
class FocusSlicePreprocessor(FullSlicePreprocessor):
    def process_patient(self, mhd_path, annotations):
        """优化版：仅处理结节所在层及相邻切片"""
        ct_scan = sitk.ReadImage(mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_scan)  # shape: (slices, height, width)
        origin = ct_scan.GetOrigin()
        spacing = ct_scan.GetSpacing()

        # 获取所有结节所在的切片索引
        target_slices = self._get_target_slices(annotations, origin, spacing, ct_array.shape[0])

        # 仅处理目标切片
        for slice_idx in target_slices:
            self.process_slice(
                ct_slice=ct_array[slice_idx],
                slice_z=origin[2] + slice_idx * spacing[2],
                spacing=spacing,
                patient_id=Path(mhd_path).stem,
                annotations=annotations,
                slice_idx=slice_idx
            )

    def _get_target_slices(self, annotations, origin, spacing, total_slices):
        """计算需要处理的切片索引"""
        # 参数配置
        around_slices = 2  # 每个结节前后各取2层
        min_slice = 0
        max_slice = total_slices - 1

        target_indices = set()

        # 遍历所有结节标注
        for _, annot in annotations.iterrows():
            # 世界坐标转体素坐标
            world_coord = [annot['coordX'], annot['coordY'], annot['coordZ']]
            voxel_z = (world_coord[2] - origin[2]) / spacing[2]
            slice_idx = int(round(voxel_z))

            # 添加目标切片及相邻层
            start = max(slice_idx - around_slices, min_slice)
            end = min(slice_idx + around_slices, max_slice)
            target_indices.update(range(start, end + 1))

        return sorted(target_indices)


# YOLOv4模型集成
class LunaYOLOv4(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        # 子类化不需要显式定义输入层
        self.backbone = CSPDarknet53()
        self.neck = PANet()

        # 检测头
        self.head_large = YOLOHead(512, len(config.anchors[0]), config.num_classes)
        self.head_medium = YOLOHead(256, len(config.anchors[1]), config.num_classes)
        self.head_small = YOLOHead(128, len(config.anchors[2]), config.num_classes)

        # 多尺度训练配置
        self.grid_sizes = config.grid_sizes
        self.anchors = config.anchors
        self.output_names = ['large', 'medium', 'small']

        self.loss_metrics = {
            'total_loss': tf.keras.metrics.Mean(name='total_loss'),
            'coord_loss': tf.keras.metrics.Mean(name='coord_loss'),
            'conf_loss': tf.keras.metrics.Mean(name='conf_loss')
        }

    def call(self, inputs, training=False):
        # 主干网络前向传播
        route_small, route_medium, route_large = self.backbone(inputs)

        # 特征金字塔融合
        x_small, x_medium, x_large = self.neck(
            (route_small, route_medium, route_large)
        )

        # 多尺度预测输出
        outputs = [
            self.head_large(x_large),       # (batch, 13, 13, 3, 5+num_classes)
            self.head_medium(x_medium),     # (batch, 26, 26, 3, 5+num_classes)
            self.head_small(x_small)        # (batch, 52, 52, 3, 5+num_classes)
        ]
        return outputs



# 处理整体的CT
if __name__ == "__main__":

    preprocessor = FullSlicePreprocessor(config)
    # 处理所有患者
    df_annot = pd.read_csv(config.annotation_csv)
    for mhd_path in glob.glob(f"{config.raw_data_dir}/*.mhd"):
        patient_id = Path(mhd_path).stem
        patient_annot = df_annot[df_annot['seriesuid'] == patient_id]
        print(f'patient_id: {patient_id}')
        preprocessor.process_patient(mhd_path, patient_annot)

    # 创建数据集
    dataset = create_dataset(config, batch_size=8)

    # 模型训练（保持原有YOLO结构不变）
    model = LunaYOLOv4(config)
    model.fit(dataset, epochs=5)

