import os
from random import random

import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 配置文件路径
DATA_DIR = 'D:/BaiduNetdiskDownload/LUNA16/subset0'
ANNOTATION_FILE = 'D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv'
TARGET_SIZE = (256, 256)

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
    """生成3D结节掩码"""
    global ANNOT_DF

    seriesuid = os.path.basename(mhd_path).split('.mhd')[0]
    nodules = ANNOT_DF[ANNOT_DF['seriesuid'] == seriesuid]

    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (z,y,x)
    mask = np.zeros_like(array, dtype=np.float32)

    for _, row in nodules.iterrows():
        try:
            # 物理坐标 -> 体素坐标 (返回x,y,z顺序)
            world_coord = [row['coordX'], row['coordY'], row['coordZ']]
            voxel_coord = image.TransformPhysicalPointToIndex(world_coord)
            z_dim, y_dim, x_dim = voxel_coord[::-1]  # 关键修复：反转顺序

            # 计算各轴半径（体素单位）
            radius = {
                'x': (row['diameter_mm'] / image.GetSpacing()[0]) / 2,
                'y': (row['diameter_mm'] / image.GetSpacing()[1]) / 2,
                'z': (row['diameter_mm'] / image.GetSpacing()[2]) / 2
            }

            # 计算椭球边界
            z_min = max(0, int(np.floor(z_dim - radius['z'])))
            z_max = min(array.shape[0], int(np.ceil(z_dim + radius['z'])))
            y_min = max(0, int(np.floor(y_dim - radius['y'])))
            y_max = min(array.shape[1], int(np.ceil(y_dim + radius['y'])))
            x_min = max(0, int(np.floor(x_dim - radius['x'])))
            x_max = min(array.shape[2], int(np.ceil(x_dim + radius['x'])))

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            # 生成椭球
            zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
            distances = (
                            ((xx - x_dim) / radius['x'])  ** 2 +
            ((yy - y_dim) / radius['y'])  ** 2 +
                                              ((zz - z_dim) / radius['z'])  ** 2
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


def load_dataset():
    """加载数据集"""
    mhd_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mhd')]

    scans, masks = [], []
    for mhd_file in mhd_files[:3]:      # 测试前3个文件
        mhd_path = os.path.join(DATA_DIR, mhd_file)
        scan_data, meta = load_scan(mhd_path)
        mask = create_mask(mhd_path)

        scans.extend(scan_data)
        masks.extend(mask)
        print(f"文件 {mhd_file} 生成的掩码中非零切片数量：{np.sum(mask.sum(axis=(1, 2, 3)) > 0)}")

    return np.array(scans), np.array(masks)


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



def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def combined_loss(alpha=0.5, gamma=2):
    def loss(y_true, y_pred):
        # Dice loss
        dice = dice_loss(y_true, y_pred)

        # Focal loss
        epsilon = 1e-5
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = -tf.reduce_mean(alpha * (1 - pt) ** gamma * tf.math.log(pt))

        return dice + focal

    return loss



# 主程序
if __name__ == "__main__":
    configure_gpu()

    # 加载数据
    scans, masks = load_dataset()
    print(f"数据维度: {scans.shape}, Mask维度: {masks.shape}")

    model = build_unet()
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=combined_loss(alpha=0.8, gamma=2))

    model.summary()
    model.fit(scans, masks, batch_size=32, epochs=20, validation_split=0.2)


    model.save("D:\PycharmProjects\classify_img\saved", save_format="tf")

    # 测试ROI处理
    sample_path = os.path.join(DATA_DIR, "D:/BaiduNetdiskDownload/LUNA16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd")  # 替换为实际路径
    scan_data, meta = load_scan(sample_path)
    annot_mask = create_mask(sample_path)
    candidate_rois = crop_roi(scan_data, annot_mask, meta)

    # 训练ROI数据
    # for roi_scan, roi_mask in candidate_rois:
    #     model.fit(np.expand_dims(roi_scan, 0), np.expand_dims(roi_mask, 0))

    # 选择包含结节的测试样本
    nonzero_indices = np.where(masks.sum(axis=(1, 2, 3)) > 0)[0]
    # test_idx = np.random.choice(nonzero_indices)



    for idx in nonzero_indices:
        # 预测并应用阈值
        pred = model.predict(scans[idx][np.newaxis, ...])
        pred_binary = (pred > 0.5).astype(np.float32)

        # 可视化
        plt.figure(figsize=(12, 6))

        # 输入图像
        plt.subplot(1, 3, 1)
        plt.imshow(scans[idx, ..., 0], cmap='gray')
        plt.title('Input')
        plt.axis('off')

        # 真实掩码
        plt.subplot(1, 3, 2)
        plt.imshow(masks[idx, ..., 0], cmap='gray', vmin=0, vmax=1)
        plt.title('Ground Truth')
        plt.axis('off')

        # 预测结果（二值化后）
        plt.subplot(1, 3, 3)
        plt.imshow(pred_binary[0, ..., 0], cmap='gray', vmin=0, vmax=1)
        plt.title('Prediction')
        plt.axis('off')

        plt.tight_layout()
        plt.show()