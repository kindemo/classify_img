import glob
import os
import math
from pathlib import Path

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



def validate_annotations(config):
    df_annot = pd.read_csv(config.annotation_csv)
    mhd_files = glob.glob(f"{config.raw_data_dir}/*.mhd")

    # 提取所有患者ID
    patient_ids = [os.path.basename(f).replace(".mhd", "") for f in mhd_files]

    # 检查标注覆盖情况
    missing = [pid for pid in patient_ids if pid not in df_annot['seriesuid'].values]
    if missing:
        print(f"警告: {len(missing)}个患者缺少标注数据，示例：{missing[:3]}")
    else:
        print("标注文件包含所有患者的标注数据")


# 输出格式 (images, (large_labels, medium_labels, small_labels))
def create_dataset(config, batch_size):
    def _parse_yolo_data(img_path):
        # 读取图像和标签（此处需确保标签与图像路径一一对应）
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)

        img = tf.image.resize(img, (config.input_size, config.input_size))
        # img = tf.expand_dims(img, axis=0)  # 添加batch维度

        img = img / 255.0  # 归一化
        print(f"处理后的图像形状: {img.shape}")
        # 应输出 (1, 416, 416, 1)

        # 生成标签路径
        label_path = tf.strings.regex_replace(img_path, "images", "labels")
        label_path = tf.strings.regex_replace(label_path, "\\.png$", ".txt")

        # 读取标签内容并解析为三个检测层的标签
        label_content = tf.io.read_file(label_path)

        def process_labels(content):
            content = content.numpy().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            # 初始化三个检测层的标签张量
            large_label = np.zeros((13, 13, 3, 6), dtype=np.float32)
            medium_label = np.zeros((26, 26, 3, 6), dtype=np.float32)
            small_label = np.zeros((52, 52, 3, 6), dtype=np.float32)

            for line in lines:
                parts = line.split()
                if len(parts) != 10:  # 确保每行10个字段
                    continue

                scale = int(parts[0])
                grid_x = int(parts[1])
                grid_y = int(parts[2])
                anchor_idx = int(parts[3])
                tx = float(parts[4])
                ty = float(parts[5])
                tw = float(parts[6])
                th = float(parts[7])
                conf = float(parts[8])
                class_id = int(parts[9])

                # 填充到对应检测层
                if scale == 0 and grid_x < 13 and grid_y < 13 and anchor_idx < 3:
                    large_label[grid_y, grid_x, anchor_idx] = [tx, ty, tw, th, conf, class_id]
                elif scale == 1 and grid_x < 26 and grid_y < 26 and anchor_idx < 3:
                    medium_label[grid_y, grid_x, anchor_idx] = [tx, ty, tw, th, conf, class_id]
                elif scale == 2 and grid_x < 52 and grid_y < 52 and anchor_idx < 3:
                    small_label[grid_y, grid_x, anchor_idx] = [tx, ty, tw, th, conf, class_id]

            return large_label, medium_label, small_label

        # 调用处理函数并设置形状
        large, medium, small = tf.py_function(
            process_labels, [label_content], [tf.float32, tf.float32, tf.float32]
        )

        # large.set_shape((13, 13, 3, 6))
        # medium.set_shape((26, 26, 3, 6))
        # small.set_shape((52, 52, 3, 6))

        # 添加形状验证
        tf.debugging.assert_shapes([
            (large, (13, 13, 3, 6)),
            (medium, (26, 26, 3, 6)),
            (small, (52, 52, 3, 6)),
        ], message="标签形状错误")

        # return img, (large, medium, small)
        return img, {
            "large": large,
            "medium": medium,
            "small": small
        }  # 改为字典形式输出

    # 构建数据集（确保返回格式为 (images, (large_labels, medium_labels, small_labels))）
    img_files = tf.data.Dataset.list_files(f"{config.processed_dir}/images/*.png")
    dataset = img_files.map(_parse_yolo_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



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





# 使用示例
if __name__ == "__main__":
    try:
        validate_annotations(config)  # 新增验证步骤

        preprocessor = LunaYoloPreprocessor(config)

        # 先加载标注文件
        df_annotations = pd.read_csv(config.annotation_csv)
        print(f"成功加载标注文件，共 {len(df_annotations)} 条记录")

        # 处理每个CT文件
        for mhd_file in glob.glob(f"{config.raw_data_dir}/*.mhd"):
            preprocessor._process_patient(mhd_file, df_annotations)

    except Exception as e:
        print(f"初始化失败: {str(e)}")

    # try:
    #     preprocessor.process_dataset()
    #     print(f"生成文件数: {preprocessor.file_counter}")
    # except Exception as e:
    #     print(f"预处理失败: {str(e)}")
    #     # 打印最后处理的文件信息
    #     if hasattr(preprocessor, 'last_processed'):
    #         print(f"最后处理的文件: {preprocessor.last_processed}")

    # 创建数据集
    train_dataset = create_dataset(config, 1)

    # 初始化模型
    model = LunaYOLOv4(config)
    model.build((None, config.input_size, config.input_size, 1))  # 显式构建输入形状
    model.compile(optimizer='adam')

    # 创建回调函数以保存模型
    checkpoint_callback = ModelCheckpoint(
        filepath=config.MODEL["model_save_path"],
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # 开始训练
    model.fit(train_dataset, epochs=15, callbacks=[checkpoint_callback])

