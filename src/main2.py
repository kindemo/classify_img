import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.draw import ellipsoid

# 配置文件路径
DATA_DIR = 'D:/BaiduNetdiskDownload/LUNA16/subset0'
ANNOTATION_FILE = 'D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv'
TARGET_SIZE = (256, 256)


def load_scan(mhd_path):
    """加载并预处理单个CT扫描"""
    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (depth, height, width)

    # 预处理
    processed = []
    for slice in array:
        slice = (slice + 1000) / 1400  # CT值归一化
        slice = np.clip(slice, 0, 1)
        slice = np.expand_dims(slice, axis=-1)  # (h, w, 1)
        slice = tf.image.resize(slice, TARGET_SIZE).numpy()
        processed.append(slice)

    return np.array(processed)  # (depth, 256, 256, 1)


def create_mask(mhd_path, df_annot):
    """生成3D结节掩码，并调整至目标尺寸"""
    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (z,y,x)
    mask = np.zeros_like(array, dtype=np.float32)

    seriesuid = os.path.basename(mhd_path).split('.mhd')[0]
    nodules = df_annot[df_annot['seriesuid'] == seriesuid]

    print(f"\n处理文件: {mhd_path}")
    print(f"找到结节数量: {len(nodules)}")

    # 获取图像参数
    spacing = image.GetSpacing()  # (x,y,z) spacing
    origin = image.GetOrigin()
    print(f"体素间距(mm): {spacing}, 原点坐标: {origin}")

    for idx, (_, row) in enumerate(nodules.iterrows(), 1):
        try:
            # 物理坐标 -> 体素坐标 (返回x,y,z顺序)
            world_coord = [row['coordX'], row['coordY'], row['coordZ']]
            voxel_coord = image.TransformPhysicalPointToIndex(world_coord)
            x_dim = voxel_coord[0]  # numpy的x轴对应SimpleITK的x
            y_dim = voxel_coord[1]  # numpy的y轴对应SimpleITK的y
            z_dim = voxel_coord[2]  # numpy的z轴对应SimpleITK的z

            print(f"\n结节 {idx}:")
            print(f"物理坐标: {world_coord}")
            print(f"体素坐标(x,y,z): {voxel_coord}")
            print(f"原始直径: {row['diameter_mm']}mm")

            # 计算各轴半径（体素单位，保留浮点精度）
            radius = {
                'x': (row['diameter_mm'] / spacing[0]) / 2,
                'y': (row['diameter_mm'] / spacing[1]) / 2,
                'z': (row['diameter_mm'] / spacing[2]) / 2
            }
            print(f"计算半径(mm): x={radius['x']:.1f}, y={radius['y']:.1f}, z={radius['z']:.1f}")

            # 计算椭球边界（向上取整，确保包含整个结节）
            z_min = int(np.floor(z_dim - radius['z']))
            z_max = int(np.ceil(z_dim + radius['z']))
            y_min = int(np.floor(y_dim - radius['y']))
            y_max = int(np.ceil(y_dim + radius['y']))
            x_min = int(np.floor(x_dim - radius['x']))
            x_max = int(np.ceil(x_dim + radius['x']))

            # 约束边界不超过图像范围
            z_min = max(0, z_min)
            z_max = min(array.shape[0], z_max)
            y_min = max(0, y_min)
            y_max = min(array.shape[1], y_max)
            x_min = max(0, x_min)
            x_max = min(array.shape[2], x_max)

            print(f"椭球范围:")
            print(f"z: [{z_min}-{z_max}], y: [{y_min}-{y_max}], x: [{x_min}-{x_max}]")

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                print("结节范围无效，跳过")
                continue

            # 生成椭网格
            zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]

            # 椭球方程 (使用浮点计算)
            distances = (
                            ((xx - x_dim) / radius['x'])  ** 2 +
            ((yy - y_dim) / radius['y'])  ** 2 +
                                              ((zz - z_dim) / radius['z'])  ** 2
            )
            ellipsoid = (distances <= 1.0).astype(np.float32)

            # 叠加到mask
            mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                mask[z_min:z_max, y_min:y_max, x_min:x_max],
                ellipsoid
            )

            print(f"添加结节区域，体素数量: {np.sum(ellipsoid)}")

        except Exception as e:
            print(f"处理结节时出错: {str(e)}")
            continue

    # 调整每层掩码尺寸
    resized_mask = []
    for z_slice in mask:
        # 使用最近邻插值保持二值性
        resized = tf.image.resize(
            np.expand_dims(z_slice, axis=-1),
            TARGET_SIZE,
            method='nearest'
        ).numpy()
        resized_mask.append(resized)

    resized_mask = np.array(resized_mask)

    # 调试输出
    total_voxels = np.sum(resized_mask > 0)
    print(f"\n掩码统计:")
    print(f"调整后尺寸: {resized_mask.shape}")
    print(f"非零体素数量: {total_voxels}")
    if total_voxels == 0:
        print("警告: 生成的掩码全黑！")
    else:
        print(f"最大像素值: {np.max(resized_mask)}, 最小像素值: {np.min(resized_mask)}")

    return resized_mask  # (depth, 256, 256, 1)



def load_dataset():
    """加载完整数据集"""
    mhd_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mhd')]

    # 正确读取CSV，假设原文件有标题行
    df_annot = pd.read_csv(ANNOTATION_FILE,
                           dtype={
                               'seriesuid': str,
                               'coordX': np.float64,
                               'coordY': np.float64,
                               'coordZ': np.float64,
                               'diameter_mm': np.float64
                           })

    scans, masks = [], []
    for mhd_file in mhd_files[:3]:  # 测试前3个文件
        mhd_path = os.path.join(DATA_DIR, mhd_file)
        scan = load_scan(mhd_path)
        mask = create_mask(mhd_path, df_annot)

        scans.extend(scan)
        masks.extend(mask)
        print(f"文件 {mhd_file} 生成的掩码中非零切片数量：{np.sum(mask.sum(axis=(1, 2, 3)) > 0)}")
    return np.array(scans), np.array(masks)


def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # 编码器 (3次下采样)
    # Block 1 (256x256)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # 128x128

    # Block 2 (128x128)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # 64x64

    # Block 3 (64x64)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # 32x32

    # 瓶颈层 (32x32)
    b = Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = Conv2D(512, 3, activation='relu', padding='same')(b)

    # 解码器 (3次上采样)
    # Up Block 1 (32x32 -> 64x64)
    u1 = UpSampling2D((2, 2))(b)
    u1 = Concatenate()([u1, c3])
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)

    # Up Block 2 (64x64 -> 128x128)
    u2 = UpSampling2D((2, 2))(u1)
    u2 = Concatenate()([u2, c2])
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)

    # Up Block 3 (128x128 -> 256x256)
    u3 = UpSampling2D((2, 2))(u2)
    u3 = Concatenate()([u3, c1])
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)

    outputs = Conv2D(1, 1, activation='sigmoid')(u3)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy')
    return model



# 主程序
if __name__ == "__main__":
    # 加载数据
    scans, masks = load_dataset()
    print(f"数据维度: {scans.shape}, Mask维度: {masks.shape}")
    print("Mask中非零像素数量:", np.sum(masks != 0))
    print("Mask像素值分布:", np.unique(masks, return_counts=True))

    # 可视化修改后代码
    non_zero_mask = masks.sum(axis=(1, 2, 3)) > 0
    print(f"总共有 {np.sum(non_zero_mask)} 个切片包含结节")

    if np.sum(non_zero_mask) > 0:
        # 显示前5个有结节的切片
        sample_indices = np.where(non_zero_mask)[0][:5]
        for idx in sample_indices:
            ct_slice = scans[idx, ..., 0]
            mask_slice = masks[idx, ..., 0]

            plt.figure(figsize=(12, 6))

            # 原始CT图像
            plt.subplot(1, 3, 1)
            plt.imshow(ct_slice, cmap='gray')
            plt.title('CT slice')
            plt.axis('off')

            # 单独显示掩码（增强对比度）
            plt.subplot(1, 3, 2)
            plt.imshow(mask_slice, cmap='gray',
                       vmin=0, vmax=1,
                       interpolation='nearest')
            plt.title('ini mask')
            plt.axis('off')

            # 叠加显示
            plt.subplot(1, 3, 3)
            plt.imshow(ct_slice, cmap='gray')
            plt.imshow(mask_slice, cmap='viridis',
                       alpha=0.3, vmin=0, vmax=1)
            plt.title('mix show')
            plt.axis('off')

            # plt.tight_layout()
            plt.show()
    else:
        print("错误：未找到任何包含结节的切片！")

    # 修改后的随机可视化代码（替换原代码末尾的随机显示部分）
    nonzero_indices = np.where(masks.sum(axis=(1, 2, 3)) > 0)[0]
    if len(nonzero_indices) > 0:
        # 随机选择最多5个非零切片
        selected = np.random.choice(nonzero_indices, size=min(5, len(nonzero_indices)), replace=False)
        for i in selected:
            plt.figure(figsize=(5, 5))
            plt.imshow(masks[i, ..., 0], cmap='gray')
            plt.title(f'Mask Slice {i}')
            plt.axis('off')
            plt.show()
    else:
        print("警告：所有掩码均为空，请检查数据生成步骤！")

    # 构建模型
    model = build_unet()
    model.summary()

    # 训练
    history = model.fit(scans, masks,
                        batch_size=8,
                        epochs=1,
                        validation_split=0.2)

    # 可视化结果
    test_idx = 100
    pred = model.predict(scans[test_idx][np.newaxis, ...])

    # 修改后的可视化部分
    plt.figure(figsize=(12, 6))

    # 显示输入图像
    plt.subplot(1, 3, 1)
    plt.imshow(scans[test_idx, ..., 0], cmap='gray')
    plt.title('Input')
    plt.axis('off')

    # 显示真实mask
    plt.subplot(1, 3, 2)
    plt.imshow(masks[test_idx, ..., 0], cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    # 显示预测结果
    plt.subplot(1, 3, 3)
    plt.imshow(pred[0, ..., 0], cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()  # 自动调整子图间距
    plt.show()