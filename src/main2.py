import os
import random

import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from src.utils import AdaptiveCombinedLoss

# 配置文件路径
DATA_DIR = 'D:/BaiduNetdiskDownload/LUNA16/subset0'
ANNOTATION_FILE = 'D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv'
TARGET_SIZE = (256, 256)
# 创建独立的验证集目录
VAL_DIR = 'D:/BaiduNetdiskDownload/LUNA16/validation_subset1'


# 全局加载标注数据
ANNOT_DF = pd.read_csv(ANNOTATION_FILE, dtype={
    'seriesuid': str,
    'coordX': np.float64,
    'coordY': np.float64,
    'coordZ': np.float64,
    'diameter_mm': np.float64
})

CAND_DF = pd.read_csv('D:/BaiduNetdiskDownload/LUNA16/CSVFILES/candidates.csv',
                      dtype={'seriesuid': str})


# 动态分配内存
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=13 * 1024)])
                print(f'Using GPU: {gpu}')
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

def load_scan(mhd_path):
    """加载并预处理CT扫描，返回数据和元数据"""
    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (depth, height, width)

    # 获取元数据
    meta = {
        'spacing': image.GetSpacing(),  # (x,y,z)
        'origin': image.GetOrigin(),  # (x,y,z)
        'seriesuid': os.path.basename(mhd_path).split('.mhd')[0]
    }

    # 预处理
    processed = []
    HU_MIN = -1000
    HU_MAX = 400

    array = np.clip(array, HU_MIN, HU_MAX)  # 先限制范围
    array = (array - HU_MIN) / (HU_MAX - HU_MIN)  # 归一化到0-1

    for slice in array:
        # slice = (slice + 1000) / 1400  # CT值归一化
        # slice = np.clip(slice, 0, 1)
        slice = np.expand_dims(slice, axis=-1)  # (h, w, 1)
        slice = tf.image.resize(slice, TARGET_SIZE).numpy()
        processed.append(slice)

    return np.array(processed), meta  # (depth, 256, 256, 1)


def create_mask(mhd_path):
    """生成3D结节掩码（修正坐标转换版本）"""
    global ANNOT_DF

    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (z,y,x)
    mask = np.zeros_like(array, dtype=np.float32)

    # 获取完整的空间元数据
    direction = image.GetDirection()  # 方向矩阵 (3x3)
    origin = image.GetOrigin()  # 物理坐标系原点
    spacing = image.GetSpacing()  # 体素间距

    # 构建物理坐标到体素坐标的转换矩阵
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(direction)
    transform.SetTranslation([origin[0], origin[1], origin[2]])
    transform.SetCenter((0, 0, 0))  # 设置旋转中心为物理坐标系原点

    seriesuid = os.path.basename(mhd_path).split('.mhd')[0]
    nodules = ANNOT_DF[ANNOT_DF['seriesuid'] == seriesuid]

    for _, row in nodules.iterrows():
        try:
            # 物理坐标 -> 连续体素坐标
            physical_coord = [row['coordX'], row['coordY'], row['coordZ']]
            continuous_coord = image.TransformPhysicalPointToContinuousIndex(physical_coord)

            # 正确坐标顺序转换 (考虑方向矩阵)
            x = continuous_coord[0]  # 体素x坐标
            y = continuous_coord[1]  # 体素y坐标
            z = continuous_coord[2]  # 体素z坐标

            # 计算各轴半径（考虑各向异性间距）
            diameter = row['diameter_mm']
            radius = [
                diameter / (spacing[0] * 2),  # x轴半径（体素单位）
                diameter / (spacing[1] * 2),  # y轴半径
                diameter / (spacing[2] * 2)  # z轴半径
            ]

            # 计算椭球边界（考虑浮点坐标）
            z_min = max(0, int(np.floor(z - radius[2])))
            z_max = min(array.shape[0], int(np.ceil(z + radius[2])))
            y_min = max(0, int(np.floor(y - radius[1])))
            y_max = min(array.shape[1], int(np.ceil(y + radius[1])))
            x_min = max(0, int(np.floor(x - radius[0])))
            x_max = min(array.shape[2], int(np.ceil(x + radius[0])))

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            # 生成椭球（使用实际体素坐标系）
            zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
            distances = (
                            ((xx - x) / radius[0])  ** 2 +
            ((yy - y) / radius[1])  ** 2 +
                                        ((zz - z) / radius[2])  ** 2
            )
            ellipsoid = (distances <= 1.0).astype(np.float32)
            mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                mask[z_min:z_max, y_min:y_max, x_min:x_max], ellipsoid)

        except Exception as e:
            print(f"处理结节时出错: {str(e)}")
            continue

    # 调整掩码尺寸
    resized_mask = []
    for z_slice in mask:
        resized = tf.image.resize(
            np.expand_dims(z_slice, axis=-1),
            TARGET_SIZE,
            method='nearest'
        ).numpy()
        resized_mask.append(resized)

    return np.array(resized_mask)  # (depth, 256, 256, 1)


def crop_roi(scan, mask, image_meta, roi_size=256):
    """生成2D ROI训练样本"""
    seriesuid = image_meta['seriesuid']
    candidates = CAND_DF[CAND_DF['seriesuid'] == seriesuid]

    spacing = image_meta['spacing']
    origin = image_meta['origin']

    rois = []
    for _, row in candidates.iterrows():
        try:
            # 物理坐标转换
            coord_z = (row['coordZ'] - origin[2]) / spacing[2]
            coord_y = (row['coordY'] - origin[1]) / spacing[1]
            coord_x = (row['coordX'] - origin[0]) / spacing[0]

            center_z = int(round(coord_z))
            center_y = int(round(coord_y))
            center_x = int(round(coord_x))

            # 确保坐标有效性
            if (center_z < 0 or center_z >= scan.shape[0] or
                    center_y < 0 or center_y >= scan.shape[1] or
                    center_x < 0 or center_x >= scan.shape[2]):
                continue

            # 计算2D边界
            y_start = max(0, center_y - roi_size // 2)
            y_end = min(scan.shape[1], center_y + roi_size // 2)
            x_start = max(0, center_x - roi_size // 2)
            x_end = min(scan.shape[2], center_x + roi_size // 2)

            # 提取并调整尺寸
            roi_scan = scan[center_z, y_start:y_end, x_start:x_end, :]
            roi_mask = mask[center_z, y_start:y_end, x_start:x_end, :]

            roi_scan = tf.image.resize(roi_scan, (roi_size, roi_size)).numpy()
            roi_mask = tf.image.resize(roi_mask, (roi_size, roi_size)).numpy()

            rois.append((roi_scan, roi_mask))

        except Exception as e:
            print(f"截取ROI出错: {str(e)}")
            continue

    return rois


def build_unet(input_shape=(256, 256, 1)):
    """构建U-Net模型"""
    inputs = Input(input_shape)

    # 编码器
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # 瓶颈层
    b = Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = Conv2D(512, 3, activation='relu', padding='same')(b)

    # 解码器
    u1 = UpSampling2D((2, 2))(b)
    u1 = Concatenate()([u1, c3])
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2))(u1)
    u2 = Concatenate()([u2, c2])
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2))(u2)
    u3 = Concatenate()([u3, c1])
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)

    outputs = Conv2D(1, 1, activation='sigmoid')(u3)

    model = Model(inputs, outputs)
    return model


def generate_slice_samples(data_dir, target_size=(256, 256)):
    """生成平衡的切片样本"""
    mhd_files = [f for f in os.listdir(data_dir) if f.endswith('.mhd')]

    # 预加载所有有效切片
    all_slices = []
    for mhd_file in mhd_files:
        mhd_path = os.path.join(data_dir, mhd_file)

        # 加载扫描和掩码
        scan_data, meta = load_scan(mhd_path)
        mask_data = create_mask(mhd_path)

        # 遍历每个切片
        for z in range(scan_data.shape[0]):
            scan_slice = scan_data[z]
            mask_slice = mask_data[z]

            # 判断是否包含结节
            has_nodule = np.any(mask_slice > 0.5)
            all_slices.append((scan_slice, mask_slice, has_nodule))

    # 分离正负样本
    positive = [x for x in all_slices if x[2]]
    negative = [x for x in all_slices if not x[2]]

    return positive, negative


def balanced_slice_generator(data_dir, batch_size=32, pos_ratio=0.5):
    """平衡切片生成器（修复版本）"""
    positive, negative = generate_slice_samples(data_dir)

    pos_batch = int(batch_size * pos_ratio)
    neg_batch = batch_size - pos_batch

    while True:
        # 随机选择样本（使用正确的random模块）
        pos_samples = random.sample(positive, pos_batch) if len(positive) >= pos_batch else []
        neg_samples = random.sample(negative, neg_batch) if len(negative) >= neg_batch else []

        # 处理样本不足的情况
        if not pos_samples and not neg_samples:
            continue
        elif not pos_samples:
            neg_samples = random.sample(negative, batch_size)
        elif not neg_samples:
            pos_samples = random.sample(positive, batch_size)

        batch = pos_samples + neg_samples
        random.shuffle(batch)  # 这里也需要使用random模块

        # 转换为numpy数组
        X = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])

        yield X, y



def augmented_generator(base_generator, datagen):
    """为生成器添加实时数据增强"""
    for X_batch, y_batch in base_generator:
        # 对每个样本进行独立增强
        aug_X = []
        aug_y = []

        for i in range(X_batch.shape[0]):
            # 合并图像和掩码进行同步增强
            combined = np.concatenate([X_batch[i], y_batch[i]], axis=-1)

            # 应用随机变换
            transformed = datagen.random_transform(combined)

            # 分割回图像和掩码
            aug_X.append(transformed[..., :1])
            aug_y.append(transformed[..., 1:2])

        yield np.array(aug_X), np.array(aug_y)


# 主程序
if __name__ == "__main__":
    configure_gpu()

    # 加载数据
    # scans, masks = load_dataset()
    # print(f"数据维度: {scans.shape}, Mask维度: {masks.shape}")

    # 构建并训练模型 (None, 256, 256, 1)
    train_gen = balanced_slice_generator(DATA_DIR, batch_size=32, pos_ratio=0.5)

    # 验证集生成器（可以单独划分验证目录）
    # val_gen = balanced_slice_generator(VAL_DIR, batch_size=16, pos_ratio=0.3)     # 独立的
    val_gen = balanced_slice_generator(DATA_DIR, batch_size=16, pos_ratio=0.3)

    model = build_unet()
    # model.compile(optimizer=Adam(learning_rate=1e-3), loss=combined_loss(alpha=0.8, gamma=2))


    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=AdaptiveCombinedLoss(initial_alpha=0.3, gamma=3.0),
        metrics=[tf.keras.metrics.Precision(name='prec'),
                 tf.keras.metrics.Recall(name='rec')]
    )

    # 添加回调监控alpha值
    class AlphaMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            alpha = self.model.loss.alpha.numpy()
            print(f"\nCurrent alpha: {alpha:.4f}")

    # 数据增强配置
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    # 创建增强后的生成器
    train_gen = augmented_generator(train_gen, datagen)

    # 训练模型
    model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=20,
        validation_data=val_gen,
        validation_steps=50
    )

    # model.summary()

    # # 测试增强生成器（测试时一般不必增强）
    # test_gen = augmented_generator(
    #     balanced_slice_generator(DATA_DIR, batch_size=4),
    #     datagen
    # )


    # 检查生成器输出
    test_gen = balanced_slice_generator(DATA_DIR, batch_size=4)
    X_batch, y_batch = next(test_gen)

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X_batch[i, ..., 0], cmap='gray')
        plt.title(f'Scan {i + 1}')
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        plt.imshow(y_batch[i, ..., 0], cmap='gray')
        plt.title(f'Mask {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



    model.save("D:\PycharmProjects\classify_img\saved", save_format="tf")
