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
from patsy import origin
from scipy import ndimage

from src.YoloLabelGenerator import YOLOLabelGenerator
from src.config import config


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
        print("调用label类")
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

        print(f'origin: {origin}, spacing: {spacing}')

        # 遍历所有切片
        for slice_idx in range(ct_array.shape[0]):
            self.process_slice(
                ct_slice=ct_array[slice_idx],
                slice_z=origin[2] + slice_idx * spacing[2],  # 当前切片的Z轴坐标
                patient_id=Path(mhd_path).stem,
                annotations=annotations,
                slice_idx=slice_idx,
                origin=self.origin,
                spacing=self.spacing
            )

    def _save_labels(self, img_path, labels):
        """安全保存标签文件"""
        label_path = self.label_dir / f"{img_path.stem}.txt"
        try:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))
            print(f"生成样本: {img_path.name} (含{len(labels)}个结节)")
        except IOError as e:
            print(f"保存失败 {label_path}: {str(e)}")
            img_path.unlink()

    def process_slice(self, ct_slice, slice_z, patient_id, annotations, slice_idx, origin, spacing):
        """智能处理切片，自动过滤无结节数据"""
        # CT值转换与图像保存
        windowed = self.apply_window(ct_slice, self.config.hu_window)
        png_img = ((windowed - windowed.min()) / (windowed.max() - windowed.min()) * 255).astype(np.uint8)

        img_name = f"{patient_id}_slice{slice_idx:04d}.png"
        img_path = self.image_dir / img_name
        cv2.imwrite(str(img_path), png_img)

        # 生成并验证标签
        labels = self._generate_labels(slice_z, annotations, ct_slice.shape, origin, spacing)

        if not labels:
            img_path.unlink()  # 删除无结节切片
            print(f"过滤无结节切片: {img_path.name}")
        else:
            self._save_labels(img_path, labels)

    def apply_window(self, image, window):
        """应用CT窗宽窗位"""
        min_val = window[0] - window[1] / 2.0
        max_val = window[0] + window[1] / 2.0
        return np.clip(image, min_val, max_val)

    def _calculate_bbox(self, annot, origin, spacing, img_w, img_h):
        """精确计算边界框（带容错机制）"""
        try:
            x_center_px = (annot['coordX'] - origin[0]) / spacing[0]
            y_center_px = (annot['coordY'] - origin[1]) / spacing[1]
            diameter_px = annot['diameter_mm'] / spacing[0]

            return (
                x_center_px / img_w,
                y_center_px / img_h,
                diameter_px / img_w,
                diameter_px / img_h  # 保持纵横比
            )
        except ZeroDivisionError:
            return (0, 0, 0, 0)

    def _validate_coordinates(self, x, y, w, h):
        """增强型坐标验证"""
        return (0 <= x <= 1 and 0 <= y <= 1 and
                w > 0 and h > 0 and
                w < 1.2 and h < 1.2)  # 允许10%的溢出容差

    def _generate_labels(self, slice_z, annotations, img_size, origin, spacing):
        """精确生成结节标签"""
        labels = []
        img_w, img_h = img_size[1], img_size[0]  # 图像宽高

        for _, annot in annotations.iterrows():
            # 空间一致性验证
            if abs(annot['coordZ'] - slice_z) > spacing[2] * 0.5:  # 半层厚度容差
                continue

            # 坐标转换双重验证
            x_center, y_center, width, height = self._calculate_bbox(annot, origin, spacing, img_w, img_h)
            if not self._validate_coordinates(x_center, y_center, width, height):
                continue

            labels.append(f"{self.config.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels

# 只关注极少量切片
class FocusSlicePreprocessor(FullSlicePreprocessor):
    def process_patient(self, mhd_path, annotations):
        """优化版：仅处理结节所在层及相邻切片"""
        ct_scan = sitk.ReadImage(mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_scan)  # shape: (slices, height, width)
        origin = ct_scan.GetOrigin()
        spacing = ct_scan.GetSpacing()
        # print(f"origin: {origin}, spacing: {spacing}")

        # 获取所有结节所在的切片索引
        target_slices = self._get_target_slices(annotations, origin, spacing, ct_array.shape[0])

        # 仅处理目标切片
        for slice_idx in target_slices:
            self.process_slice(
                ct_slice=ct_array[slice_idx],
                slice_z=origin[2] + slice_idx * spacing[2],
                patient_id=Path(mhd_path).stem,
                annotations=annotations,
                slice_idx=slice_idx,
                origin=origin,
                spacing=spacing
            )

    def _get_target_slices(self, annotations, origin, spacing, total_slices):
        """精确获取结节所在切片索引"""
        target_indices = set()

        for _, annot in annotations.iterrows():
            # 世界坐标系转体素坐标系
            world_z = annot['coordZ']
            voxel_z = (world_z - origin[2]) / spacing[2]
            slice_idx = int(round(voxel_z))

            # 验证切片有效性
            if 0 <= slice_idx < total_slices:
                target_indices.add(slice_idx)

        return sorted(target_indices)






# 处理整体的CT
if __name__ == "__main__":

    preprocessor = FocusSlicePreprocessor(config)
    # 处理所有患者
    df_annot = pd.read_csv(config.annotation_csv)
    for mhd_path in glob.glob(f"{config.raw_data_dir}/*.mhd"):
        patient_id = Path(mhd_path).stem
        patient_annot = df_annot[df_annot['seriesuid'] == patient_id]
        # print(f'patient_id: {patient_id}')
        preprocessor.process_patient(mhd_path, patient_annot)

    # # 创建数据集
    # dataset = create_dataset(config, batch_size=8)
    #
    # # 模型训练（保持原有YOLO结构不变）
    # model = LunaYOLOv4(config)
    # model.fit(dataset, epochs=5)

