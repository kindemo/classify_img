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
    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (z,y,x)
    mask = np.zeros_like(array, dtype=np.float32)

    seriesuid = os.path.basename(mhd_path).split('.mhd')[0]
    nodules = df_annot[df_annot['seriesuid'] == seriesuid]

    for _, row in nodules.iterrows():
        try:
            world_coord = [row['coordX'], row['coordY'], row['coordZ']]
            voxel_coord = image.TransformPhysicalPointToIndex(world_coord)
            x, y, z = voxel_coord  # SimpleITK顺序(x,y,z)

            # 转换为numpy数组的(z,y,x)顺序
            z_dim = z
            y_dim = y
            x_dim = x

            spacing = image.GetSpacing()
            diameter = row['diameter_mm']

            # 计算半径（体素单位）
            radius_x = (diameter / spacing[0]) / 2
            radius_y = (diameter / spacing[1]) / 2
            radius_z = (diameter / spacing[2]) / 2

            # 强制最小半径（关键！）
            radius_x = max(2, int(round(radius_x)))  # 至少2像素
            radius_y = max(2, int(round(radius_y)))
            radius_z = max(1, int(round(radius_z)))  # z轴可接受1层

            # 计算椭球范围
            z_min = max(0, z_dim - radius_z)
            z_max = min(array.shape[0], z_dim + radius_z + 1)
            y_min = max(0, y_dim - radius_y)
            y_max = min(array.shape[1], y_dim + radius_y + 1)
            x_min = max(0, x_dim - radius_x)
            x_max = min(array.shape[2], x_dim + radius_x + 1)

            # 生成椭球
            zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
            distances = (
                    ((xx - x_dim) / radius_x) ** 2 +
                    ((yy - y_dim) / radius_y) ** 2 +
                    ((zz - z_dim) / radius_z) ** 2
            )
            mask_roi = (distances <= 1.0).astype(np.float32)

            # 写入mask
            mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                mask[z_min:z_max, y_min:y_max, x_min:x_max],
                mask_roi
            )

        except Exception as e:
            print(f"处理结节时出错: {e}")
            continue

    # 调整尺寸
    resized_mask = [
        tf.image.resize(np.expand_dims(s, -1), TARGET_SIZE, method='nearest').numpy()
        for s in mask
    ]
    return np.array(resized_mask)

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

    # 随机可视化5个真实mask
    for i in np.random.randint(0, len(masks), 5):
        plt.imshow(masks[i, ..., 0], cmap='gray')
        plt.title(f'Mask {i}')
        plt.show()

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