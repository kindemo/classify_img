import os
import random

import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from src.utils import AdaptiveCombinedLoss
# import albumentations as A

# 配置文件路径
DATA_DIR = 'D:/BaiduNetdiskDownload/LUNA16/'
ANNOTATION_FILE = 'D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv'
TARGET_SIZE = (256, 256)
# 创建独立的验证集目录
# VAL_DIR = 'D:/BaiduNetdiskDownload/LUNA16/validation_subset9'


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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=13 * 1024)]
        )
    except RuntimeError as e:
        print(e)


def load_scan(mhd_path):
    """
        加载并预处理CT扫描，返回数据和元数据
        输出(depth, 256, 256, 1)
    """
    image = sitk.ReadImage(mhd_path)
    processed = []

    HU_MIN = -1000.0
    HU_MAX = 400.0

    # 获取元数据
    meta = {
        'spacing': image.GetSpacing(),  # (x,y,z)
        'origin': image.GetOrigin(),  # (x,y,z)
        'seriesuid': os.path.basename(mhd_path).split('.mhd')[0]
    }

    for z in range(image.GetDepth()):
        # 逐层读取切片
        array = sitk.GetArrayFromImage(image[:, :, z]).astype(np.float32)
        array = np.clip(array, HU_MIN, HU_MAX)
        array = (array - HU_MIN) / (HU_MAX - HU_MIN)
        # 调整尺寸并保存
        resized = tf.image.resize(np.expand_dims(array, -1), TARGET_SIZE).numpy()
        processed.append(resized)

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


def crop_roi(scan_data, mask_data, image_meta, roi_size=128, max_samples=1000):
    """修正后的ROI截取函数（包含坐标验证和样本平衡）"""
    seriesuid = image_meta['seriesuid']
    candidates = CAND_DF[CAND_DF['seriesuid'] == seriesuid]

    # 加载原始图像以获取方向信息
    mhd_path = os.path.join(DATA_DIR, 'subset0', f'{seriesuid}.mhd')  # 根据实际路径调整
    image = sitk.ReadImage(mhd_path)

    samples = []
    true_positives = 0
    false_positives = 0

    for _, row in candidates.iterrows():
        # 物理坐标 -> 连续体素坐标（考虑方向矩阵）
        try:
            physical_point = (row['coordX'], row['coordY'], row['coordZ'])
            continuous_index = image.TransformPhysicalPointToContinuousIndex(physical_point)
        except:
            continue

        # 转换坐标顺序：SimpleITK返回(x,y,z)，numpy数组是(z,y,x)
        z = int(round(continuous_index[2]))  # 深度维度
        y = int(round(continuous_index[1]))  # 高度
        x = int(round(continuous_index[0]))  # 宽度

        # 验证坐标有效性
        if (z < 0 or z >= scan_data.shape[0] or
                y < 0 or y >= scan_data.shape[1] or
                x < 0 or x >= scan_data.shape[2]):
            continue

        # 确定是否为真实结节（3mm容差）
        is_true = False
        diameter = 3.0  # 候选框默认直径
        for _, nodule in ANNOT_DF[ANNOT_DF['seriesuid'] == seriesuid].iterrows():
            distance = np.sqrt(
                (nodule['coordX'] - row['coordX']) ** 2 +
                (nodule['coordY'] - row['coordY']) ** 2 +
                (nodule['coordZ'] - row['coordZ']) ** 2
            )
            if distance < max(nodule['diameter_mm'], diameter):
                is_true = True
                diameter = nodule['diameter_mm']
                break

        # 平衡采样
        if is_true:
            if true_positives >= max_samples // 2:  # 控制正样本数量
                continue
            true_positives += 1
        else:
            if false_positives >= max_samples // 2:  # 控制负样本数量
                continue
            false_positives += 1

        # 计算ROI边界（考虑不同尺寸候选框）
        radius = int(round(diameter / (image_meta['spacing'][0] * 2)))  # 基于X方向间距
        size = min(roi_size, radius * 2)

        # 确保边界不越界
        y_start = max(0, y - size // 2)
        y_end = min(scan_data.shape[1], y + size // 2)
        x_start = max(0, x - size // 2)
        x_end = min(scan_data.shape[2], x + size // 2)

        # 截取扫描和掩码区域
        roi_scan = scan_data[z, y_start:y_end, x_start:x_end, :]
        roi_mask = mask_data[z, y_start:y_end, x_start:x_end, :]

        # 调整尺寸并标准化
        roi_scan = tf.image.resize(roi_scan, (roi_size, roi_size)).numpy()
        roi_mask = tf.image.resize(roi_mask, (roi_size, roi_size)).numpy()

        # 增强负样本（添加随机偏移）
        if not is_true and np.random.rand() > 0.5:
            offset = np.random.randint(-5, 5, size=2)
            roi_scan = np.roll(roi_scan, offset, axis=(0, 1))

        samples.append((roi_scan, roi_mask, int(is_true)))

    return samples

def build_model():
    base = EfficientNetB0(weights=None, include_top=False, input_shape=(256,256,1))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=outputs)

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


# def generate_slice_samples(data_dir):
#     subset_dirs = [d for d in os.listdir(data_dir) if d.startswith("subset")]
#
#     for subset_dir in subset_dirs:
#         subset_path = os.path.join(data_dir, subset_dir)
#         mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]
#
#         for mhd_file in mhd_files:
#             mhd_path = os.path.join(subset_path, mhd_file)
#
#             # 流式处理每个CT文件
#             scan_data, meta = load_scan(mhd_path)
#             mask_data = create_mask(mhd_path)
#
#             for z in range(scan_data.shape[0]):
#                 scan_slice = scan_data[z]
#                 mask_slice = mask_data[z]
#                 has_nodule = np.any(mask_slice > 0.5)
#                 yield (scan_slice, mask_slice, has_nodule)


def create_single_slice_mask(image, z_index, meta):
    """生成单层结节掩码"""
    spacing = meta['spacing']
    origin = meta['origin']
    direction = meta['direction']
    seriesuid = meta['seriesuid']

    # 初始化单层掩码
    array = sitk.GetArrayFromImage(image[:, :, z_index])
    mask = np.zeros_like(array, dtype=np.float32)

    # 获取当前切片对应的物理坐标范围
    slice_origin = [
        origin[0] + z_index * direction[2] * spacing[0],
        origin[1] + z_index * direction[5] * spacing[1],
        origin[2] + z_index * direction[8] * spacing[2]
    ]

    # 查询当前切片内的结节
    nodules = ANNOT_DF[ANNOT_DF['seriesuid'] == seriesuid]
    for _, row in nodules.iterrows():
        # 坐标转换（优化后的版本）
        physical_coord = [row['coordX'], row['coordY'], row['coordZ']]
        try:
            # 计算与当前切片的距离
            z_distance = abs(physical_coord[2] - slice_origin[2])
            if z_distance > row['diameter_mm'] / 2:
                continue

            # 投影到当前切片
            coord_2d = image.TransformPhysicalPointToIndex(physical_coord)
            y, x = coord_2d[1], coord_2d[0]

            # 生成2D圆形掩码
            diameter = row['diameter_mm']
            radius = diameter / (spacing[0] * 2)  # 使用X方向间距
            xx, yy = np.mgrid[:array.shape[0], :array.shape[1]]
            distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            circle = (distance <= radius).astype(np.float32)

            mask = np.maximum(mask, circle)

        except Exception as e:
            continue

    # 调整尺寸并返回
    return tf.image.resize(np.expand_dims(mask, -1), TARGET_SIZE, method='nearest').numpy()


def generate_slice_samples(data_dir):
    subset_dirs = [d for d in os.listdir(data_dir) if d.startswith("subset")]

    for subset_dir in subset_dirs:
        subset_path = os.path.join(data_dir, subset_dir)
        mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]

        for mhd_file in mhd_files:
            mhd_path = os.path.join(subset_path, mhd_file)
            image = sitk.ReadImage(mhd_path)
            meta = {
                'spacing': image.GetSpacing(),
                'origin': image.GetOrigin(),
                'direction': image.GetDirection(),
                'seriesuid': os.path.basename(mhd_path).split('.mhd')[0]
            }

            for z in range(image.GetDepth()):
                # 逐层读取，避免加载整个3D数据
                scan_slice = sitk.GetArrayFromImage(image[:, :, z]).astype(np.float32)
                scan_slice = np.clip(scan_slice, -1000, 400)
                scan_slice = (scan_slice + 1000) / 1400  # 标准化

                # 调整尺寸（使用CPU）
                with tf.device('/cpu:0'):
                    scan_slice = tf.image.resize(
                        np.expand_dims(scan_slice, -1),
                        TARGET_SIZE
                    ).numpy()

                # 生成掩码
                mask_slice = create_single_slice_mask(image, z, meta)
                has_nodule = np.any(mask_slice > 0.5)

                yield (scan_slice, mask_slice, has_nodule)

                # 及时释放内存
                del scan_slice, mask_slice


def balanced_slice_generator(data_dir, batch_size=32, pos_ratio=0.5):
    pos_buffer, neg_buffer = [], []
    buffer_size = 1000  # 控制内存占用的缓冲区大小

    while True:
        # 动态填充缓冲区
        while len(pos_buffer) < batch_size * pos_ratio or len(neg_buffer) < batch_size * (1 - pos_ratio):
            scan, mask, has_nodule = next(generate_slice_samples(data_dir))
            if has_nodule:
                pos_buffer.append((scan, mask))
            else:
                neg_buffer.append((scan, mask))

        # 从缓冲区采样
        pos_samples = random.sample(pos_buffer, int(batch_size * pos_ratio))
        neg_samples = random.sample(neg_buffer, batch_size - int(batch_size * pos_ratio))

        # 组装batch
        batch = pos_samples + neg_samples
        random.shuffle(batch)

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


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice = (2*|X ∩ Y|) / (|X| + |Y|)
         = 2*TP / (2*TP + FP + FN)
    """
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# 主程序
if __name__ == "__main__":

    # 加载数据
    # scans, masks = load_dataset()
    # print(f"数据维度: {scans.shape}, Mask维度: {masks.shape}")

    # 构建并训练模型 (None, 256, 256, 1)
    train_gen = balanced_slice_generator(DATA_DIR, batch_size=16, pos_ratio=0.5)

    # 验证集生成器（可以单独划分验证目录）
    # val_gen = balanced_slice_generator(VAL_DIR, batch_size=4, pos_ratio=0.3)     # 独立的
    # val_gen = balanced_slice_generator(DATA_DIR, batch_size=16, pos_ratio=0.3)
    val_gen = train_gen     # 先测试为一样的

    # model = build_model()       # 使用更先进的模型结构
    model = build_unet()
    # model.compile(optimizer=Adam(learning_rate=1e-3), loss=combined_loss(alpha=0.8, gamma=2))


    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=AdaptiveCombinedLoss(initial_alpha=0.3, gamma=3.0),
        metrics=[dice_coefficient,
        tf.keras.metrics.Precision(name='prec'),
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
    # 创建增强后的生成器 （先不增强）
    # train_gen = augmented_generator(train_gen, datagen)

    # 训练模型
    model.fit(
        train_gen,
        steps_per_epoch=8,
        epochs=3,
        validation_data=val_gen,
        validation_steps=4
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
