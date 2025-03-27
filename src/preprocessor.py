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
from scipy import ndimage

from src.yolo_model import CSPDarknet53, PANet, YOLOHead


# 配置文件
class LunaConfig:
    def __init__(self):
        # 数据路径
        self.raw_data_dir = "D:\BaiduNetdiskDownload\LUNA16\subset0"
        self.processed_dir = "D:\BaiduNetdiskDownload\LUNA16\processed"
        self.annotation_csv = "D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv"

        # 验证路径
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {self.raw_data_dir}")
        if not os.path.exists(self.annotation_csv):
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_csv}")


        self.spacing = (1.0, 1.0, 1.0)  # 添加空间间距参数

        # YOLO格式配置
        self.img_size = (416, 416)  # 输入图像尺寸
        self.grid_sizes = [52, 26, 13]  # YOLO特征图尺寸
        self.anchors = [
            [(12, 16), (19, 36), (40, 28)],  # 小尺度
            [(36, 75), (76, 55), (72, 146)],  # 中尺度
            [(142, 110), (192, 243), (459, 401)]  # 大尺度
        ]

        # 预处理参数
        self.target_spacing = 1.0  # 体素标准化间距(mm)
        self.cube_size = 32  # 截取立方体尺寸
        self.hu_range = (-1000, 400)  # HU值截断范围


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


# LUNA16预处理类
class LunaYoloPreprocessor:
    def __init__(self, config):
        self.config = config
        self._create_dirs()
        self.file_counter = 0  # 新增文件计数器

    def _create_dirs(self):
        """安全创建目录（兼容Windows）"""
        images_dir = Path(self.config.processed_dir) / "images"
        labels_dir = Path(self.config.processed_dir) / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 验证目录可写性
        test_file = images_dir / "test.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise PermissionError(f"目录不可写: {images_dir}，错误: {str(e)}")

    def _process_patient(self, mhd_path, df_annotations):
        """统一处理单个患者的CT扫描和标注数据"""
        # 规范患者ID格式
        patient_id = os.path.basename(mhd_path).replace(".mhd", "").strip()
        print(f"\n正在处理患者: {patient_id}")

        try:
            # 加载CT扫描数据
            ct_scan = self._load_ct_scan(mhd_path)
        except Exception as e:
            print(f"CT扫描加载失败: {str(e)}")
            return

        # 统一获取标注数据（带空格处理）
        if 'seriesuid' not in df_annotations.columns:
            raise KeyError("标注文件中缺少seriesuid列")

        # 精确匹配标注（带空格处理）
        patient_annots = df_annotations[
            df_annotations['seriesuid'].str.strip() == patient_id
            ]

        # 标注存在性检查
        if patient_annots.empty:
            print(f"警告: 患者 {patient_id} 无有效标注")
            print(f"标注示例: {df_annotations['seriesuid'].iloc[0]}")
            return

        print(f"发现 {len(patient_annots)} 个有效结节标注")

        # 处理每个结节标注
        for idx, annot in patient_annots.iterrows():
            try:
                # 添加进度显示
                print(f"处理结节 {idx + 1}/{len(patient_annots)}", end='\r')
                self._process_nodule(ct_scan, annot, patient_id)
            except Exception as e:
                print(f"\n结节处理失败: {str(e)}")
                continue

        print(f"\n患者 {patient_id} 处理完成")

    def _generate_yolo_label(self, img_shape, annot, img_path):
        """生成YOLO标注文件（增加参数验证）"""
        # 参数验证
        if not Path(img_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        if img_shape[0] <= 0 or img_shape[1] <= 0:
            raise ValueError(f"无效的图像尺寸: {img_shape}")

        # 坐标转换（添加边界检查）
        x_center = 0.5
        y_center = 0.5
        width = np.clip(annot['diameter_mm'] / (self.config.cube_size * self.config.spacing[0]), 0, 1)
        height = np.clip(annot['diameter_mm'] / (self.config.cube_size * self.config.spacing[1]), 0, 1)

        # 构建标注内容
        label_line = f"{scale} {x_center} {y_center} {width} {height}\n"

        # 保存标注
        label_path = Path(img_path).parent.parent / "labels" / Path(img_path).name.replace(".png", ".txt")
        label_path.write_text(label_line)
        return label_path

    def _load_ct_scan(self, mhd_path):
        """加载CT扫描（添加详细错误信息）"""
        try:
            itk_image = sitk.ReadImage(str(mhd_path))  # 确保路径为字符串
            img_array = sitk.GetArrayFromImage(itk_image)

            # 添加CT元数据验证
            if img_array.size == 0:
                raise ValueError("CT数据为空")

            # 标准化处理
            img_array = np.clip(img_array, *self.config.hu_range)
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            return {
                'data': img_array,
                'origin': itk_image.GetOrigin(),
                'spacing': itk_image.GetSpacing()
            }
        except Exception as e:
            print(f"加载CT文件失败: {mhd_path}")
            raise

    def process_dataset(self):
        """处理整个数据集（添加进度跟踪）"""
        try:
            df_annotations = pd.read_csv(self.config.annotation_csv)
            total_files = len(glob.glob(f"{self.config.raw_data_dir}/*.mhd"))

            for idx, mhd_file in enumerate(glob.glob(f"{self.config.raw_data_dir}/*.mhd")):
                patient_id = os.path.basename(mhd_file).split('.')[0]
                print(f"Processing {idx + 1}/{total_files}: {patient_id}")
                self._process_patient(mhd_file, patient_id, df_annotations)

            print(f"预处理成功完成! 生成文件保存在: {self.config.processed_dir}")
        except Exception as e:
            print(f"预处理失败: {str(e)}")
            raise

    def _process_nodule(self, ct_scan, annot, patient_id):
        """处理单个结节"""
        # 坐标转换
        world_coord = np.array([annot['coordX'], annot['coordY'], annot['coordZ']])
        voxel_coord = self._world_to_voxel(world_coord, ct_scan['origin'], ct_scan['spacing'])

        # 截取立方体
        cube = self._extract_cube(ct_scan['data'], voxel_coord)

        # 生成多尺度数据
        for scale in range(3):
            self._generate_scale_data(cube, annot, patient_id, scale)

    def _world_to_voxel(self, world_coord, origin, spacing):
        """世界坐标转体素坐标"""
        return (world_coord - origin) / spacing

    def _extract_cube(self, img_3d, center_voxel):
        """截取3D立方体"""
        size = self.config.cube_size
        z, y, x = center_voxel.astype(int)

        # 边界保护
        z_start = max(z - size // 2, 0)
        y_start = max(y - size // 2, 0)
        x_start = max(x - size // 2, 0)

        return img_3d[z_start:z_start + size, y_start:y_start + size, x_start:x_start + size]

    # 修改后的_generate_scale_data方法
    def _generate_scale_data(self, cube, annot, patient_id, scale):
        """生成不同尺度的训练数据（修复3D缩放问题）"""
        target_size = self.config.grid_sizes[scale]

        # 直接进行三维缩放（无需预初始化）
        zoom_factor = [
            target_size / cube.shape[0],
            target_size / cube.shape[1],
            target_size / cube.shape[2]
        ]

        scaled_cube = ndimage.zoom(cube, zoom_factor, order=1)

        # 保存图像切片
        for z in range(scaled_cube.shape[0]):
            img_slice = scaled_cube[z]
            img_path = f"{self.config.processed_dir}/images/{patient_id}_s{scale}_z{z}.png"
            cv2.imwrite(img_path, img_slice)
            self._create_yolo_label(img_path, annot, scale)

    def _create_yolo_label(self, img_path, annot, scale):
        """生成YOLO标注文件"""
        # 计算归一化坐标
        diameter = annot['diameter_mm']
        x_center = 0.5  # 立方体中心
        y_center = 0.5
        width = diameter / (self.config.cube_size * self.config.spacing[0])
        height = diameter / (self.config.cube_size * self.config.spacing[1])

        # 构建标注内容
        label_line = f"{scale} {x_center} {y_center} {width} {height}\n"

        # 保存标注
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        with open(label_path, 'w') as f:
            f.write(label_line)

    def _save_slice_image(self, slice_img, patient_id, z_index):
        # 使用pathlib处理路径
        save_dir = Path(self.config.processed_dir) / "images"
        save_dir.mkdir(exist_ok=True)

        filename = f"{patient_id}_z{z_index}.png"
        path = save_dir / filename

        # 验证图像数据
        if slice_img.shape != (self.config.cube_size, self.config.cube_size):
            raise ValueError(f"图像尺寸错误: {slice_img.shape}")

        # 保存前打印调试信息
        print(f"保存图像到: {path.absolute()}")
        print(f"图像数据范围: {slice_img.min()} - {slice_img.max()}")

        cv2.imwrite(str(path), slice_img)

        # 验证文件确实存在
        if not path.exists():
            raise RuntimeError(f"文件保存失败: {path}")
        return path


# YOLOv4模型集成
class LunaYOLOv4(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.backbone = CSPDarknet53()
        self.neck = PANet()
        self.heads = [YOLOHead(512, 3, 5),
                      YOLOHead(256, 3, 5),
                      YOLOHead(128, 3, 5)]

        # 多尺度训练配置
        self.grid_sizes = config.grid_sizes
        self.anchors = config.anchors
        self.output_names = ['large', 'medium', 'small']

    def call(self, inputs):
        # 前向传播
        route_small, route_medium, route_large = self.backbone(inputs)
        x_small, x_medium, x_large = self.neck((route_small, route_medium, route_large))

        # 多尺度输出
        outputs = [
            self.heads[0](x_large),  # 大尺度检测
            self.heads[1](x_medium),  # 中尺度检测
            self.heads[2](x_small)  # 小尺度检测
        ]
        return outputs


# 数据管道
def create_dataset(config, batch_size=8):
    def _parse_yolo_data(img_path):
        # 读取图像
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, config.img_size)

        # 生成标签路径（使用两次正则替换）
        label_path = tf.strings.regex_replace(img_path, "images", "labels")
        label_path = tf.strings.regex_replace(label_path, "\\.png$", ".txt")  # 精确匹配.png结尾

        # 读取标注文件
        label = tf.io.read_file(label_path)
        parts = tf.strings.split(label)

        # 转换为数值类型
        return img / 255.0, (
            tf.strings.to_number(parts[1:]),  # large层标签
            tf.strings.to_number(parts[1:]),  # medium层标签
            tf.strings.to_number(parts[1:])  # small层标签
        )

    # 创建数据集
    img_files = tf.data.Dataset.list_files(f"{config.processed_dir}/images/*.png")
    return img_files.map(_parse_yolo_data).batch(batch_size)


# 使用示例
if __name__ == "__main__":
    try:
        config = LunaConfig()
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
    train_dataset = create_dataset(config)

    # 初始化模型
    model = LunaYOLOv4(config)
    model.compile(optimizer='adam')

    # 开始训练
    model.fit(train_dataset, epochs=50)