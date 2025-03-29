import glob
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from src.config import config  # 确保config模块正确配置


def world_to_voxel(world_coord, origin, spacing):
    """将世界坐标转换为体素坐标"""
    return (world_coord - origin) / spacing


def process_mhd_to_yolo(mhd_path, df_annot):
    """处理单个MHD文件并转换为YOLO格式"""
    try:
        # 加载CT图像元数据
        itk_image = sitk.ReadImage(mhd_path)
        origin = np.array(itk_image.GetOrigin(), dtype=np.float32)
        spacing = np.array(itk_image.GetSpacing(), dtype=np.float32)
        size = np.array(itk_image.GetSize(), dtype=np.int32)  # 获取三维尺寸 (x, y, z)

        # 验证元数据有效性
        if any(spacing <= 0):
            raise ValueError(f"Invalid spacing value in {mhd_path}")
        if any(size <= 0):
            raise ValueError(f"Invalid image size in {mhd_path}")

        # 提取seriesuid（兼容带/不带空格的情况）
        seriesuid = os.path.basename(mhd_path).replace('.mhd', '').strip()

        # 获取对应标注
        patient_annots = df_annot[df_annot['seriesuid'].str.strip() == seriesuid]
        if len(patient_annots) == 0:
            return []

        results = []
        for _, annot in patient_annots.iterrows():
            # 坐标有效性检查
            try:
                world_coord = np.array([
                    float(annot['coordX']),
                    float(annot['coordY']),
                    float(annot['coordZ'])
                ])
                diameter_mm = float(annot['diameter_mm'])
            except (ValueError, KeyError) as e:
                print(f"Invalid annotation format in {seriesuid}: {e}")
                continue

            # 坐标转换
            voxel_coord = world_to_voxel(world_coord, origin, spacing)
            x, y, z = voxel_coord

            # Z轴切片有效性检查
            z_index = int(round(z))
            if not (0 <= z_index < size[2]):
                print(f"Slice index {z_index} out of range [0-{size[2] - 1}] in {seriesuid}")
                continue

            # 计算YOLO格式参数
            try:
                # 计算像素尺寸（考虑各向异性）
                pixel_size_x = diameter_mm / spacing[0]
                pixel_size_y = diameter_mm / spacing[1]

                # 归一化处理（添加微小量防止除零）
                x_center = x / (size[0] - 1e-8)
                y_center = y / (size[1] - 1e-8)
                width = pixel_size_x / (size[0] - 1e-8)
                height = pixel_size_y / (size[1] - 1e-8)

                # 边界检查（YOLO要求0-1之间）
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    print(f"坐标超出范围 in {seriesuid}: x={x_center:.4f}, y={y_center:.4f}")
                    continue

                results.append({
                    'seriesuid': seriesuid,
                    'x_center': round(x_center, 6),
                    'y_center': round(y_center, 6),
                    'width': round(width, 6),
                    'height': round(height, 6)
                })
            except Exception as e:
                print(f"Error processing annotation in {seriesuid}: {e}")

        return results

    except Exception as e:
        print(f"Error processing {mhd_path}: {e}")
        return []


if __name__ == "__main__":

    # 加载标注数据
    df_annot = pd.read_csv(config.annotation_csv)

    # 处理所有CT扫描
    all_results = []
    for mhd_path in glob.glob(os.path.join(config.raw_data_dir, "*.mhd")):
        results = process_mhd_to_yolo(mhd_path, df_annot)
        all_results.extend(results)
        print(f"Processed {mhd_path}: found {len(results)} nodules")

    # 保存结果
    df_output = pd.DataFrame(all_results)
    df_output.to_excel(config.output_excel_path, index=False)
    print(f"转换完成，共处理{len(all_results)}个结节。保存路径：{config.output_excel_path}")