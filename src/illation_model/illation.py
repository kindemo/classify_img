import tensorflow as tf
from pathlib import Path

from matplotlib import pyplot as plt

# # 加载导出的模型（包含预处理）
# model_path = "D:/PycharmProjects/classify_img/saved"
# loaded_model = tf.saved_model.load(model_path)
#
#
# data_dir = Path("D:/PycharmProjects/wark_by_voice")
# audio_file_path = str(data_dir / 'miya_long3.wav')
#
#
# # 直接调用模型处理输入（传入文件路径）
# input_data = tf.constant(audio_file_path, dtype=tf.string)
# output = loaded_model(input_data)


# test_model.py
import os
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt
from main2 import dice_loss

# 配置文件路径
TEST_DATA_DIR = 'D:/BaiduNetdiskDownload/LUNA16/subset0'  # 替换为测试数据路径
MODEL_PATH = '/saved'  # 替换为模型保存路径
TARGET_SIZE = (256, 256)


def load_test_scan(mhd_path):
    """加载并预处理CT扫描（测试专用）"""
    image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(image)  # (depth, height, width)

    # 预处理参数
    HU_MIN = -1000
    HU_MAX = 400

    array = np.clip(array, HU_MIN, HU_MAX)
    array = (array - HU_MIN) / (HU_MAX - HU_MIN)

    processed = []
    for slice in array:
        slice = np.expand_dims(slice, axis=-1)
        slice = tf.image.resize(slice, TARGET_SIZE).numpy()
        processed.append(slice)

    return np.array(processed)  # (depth, 256, 256, 1)


def visualize_prediction(scan, pred_mask, slice_idx):
    """可视化预测结果"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(scan[slice_idx, ..., 0], cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask[slice_idx, ..., 0], cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted Nodules')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载模型
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'dice_loss': dice_loss}
    )
    print("模型加载成功，输入规格：", model.input_shape)

    # 加载测试数据
    test_file = "1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd"  # 替换为实际文件名
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    scan_data = load_test_scan(test_path)
    print("测试数据维度：", scan_data.shape)

    # 进行预测
    pred_masks = model.predict(scan_data)
    pred_masks = (pred_masks > 0.5).astype(np.float32)  # 二值化

    # 可视化随机切片
    slice_idx = np.random.randint(0, scan_data.shape[0])
    visualize_prediction(scan_data, pred_masks, slice_idx)