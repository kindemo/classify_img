# batch_test_model.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.sample.illation import load_test_scan


# 配置参数
TEST_DIR = 'D:/BaiduNetdiskDownload/LUNA16/subset0'  # 测试集目录
MODEL_PATH = '/saved'
OUTPUT_DIR = '../predictions'  # 预测结果存储目录
BATCH_SIZE = 8
WORKERS = 4  # 并行处理进程数

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def dice_coefficient(y_true, y_pred):
    """计算3D Dice系数"""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


def process_single_file(args):
    """单文件处理函数，适用于并行处理"""
    mhd_path, model = args
    try:
        # 加载数据
        scan = load_test_scan(mhd_path)
        series_uid = os.path.basename(mhd_path).split('.')[0]

        # 预测
        pred = model.predict(scan, verbose=0)
        pred_binary = (pred > 0.5).astype(np.float32)

        # 保存结果
        np.save(os.path.join(OUTPUT_DIR, f'{series_uid}_pred.npy'), pred_binary)

        # 若有真实标注则计算指标
        mask_path = os.path.join(TEST_DIR, f'{series_uid}.npy')  # 假设已保存真实mask
        if os.path.exists(mask_path):
            true_mask = np.load(mask_path)
            dice = dice_coefficient(true_mask, pred_binary)
            return {'seriesuid': series_uid, 'dice': dice}

        return {'seriesuid': series_uid}

    except Exception as e:
        print(f"处理文件 {os.path.basename(mhd_path)} 失败: {str(e)}")
        return None


def generate_report(result_list):
    """生成评估报告"""
    df = pd.DataFrame([r for r in result_list if r is not None])

    if 'dice' in df.columns:
        print("\n性能报告：")
        print(f"平均Dice系数: {df['dice'].mean():.4f}")
        print(f"最大Dice: {df['dice'].max():.4f}")
        print(f"最小Dice: {df['dice'].min():.4f}")

    report_path = os.path.join(OUTPUT_DIR, 'performance_report.csv')
    df.to_csv(report_path, index=False)
    print(f"\n完整报告已保存至：{report_path}")


def visualize_sample_results(n_samples=3):
    """可视化随机样本结果"""
    pred_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_pred.npy')]
    selected = np.random.choice(pred_files, min(n_samples, len(pred_files)), replace=False)

    for f in selected:
        series_uid = f.split('_pred')[0]
        pred = np.load(os.path.join(OUTPUT_DIR, f))
        scan_path = os.path.join(TEST_DIR, f'{series_uid}.mhd')
        scan = load_test_scan(scan_path)

        # 随机选择三个层面
        slices = np.random.choice(pred.shape[0], 3, replace=False)

        plt.figure(figsize=(15, 10))
        for i, s in enumerate(slices, 1):
            plt.subplot(2, 3, i)
            plt.imshow(scan[s, ..., 0], cmap='gray')
            plt.title(f'原始切片 {s}')
            plt.axis('off')

            plt.subplot(2, 3, i + 3)
            plt.imshow(pred[s, ..., 0], cmap='jet', alpha=0.5)
            plt.title('预测结果')
            plt.axis('off')

        plt.suptitle(f'Series: {series_uid}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{series_uid}_vis.png'))
        plt.close()


if __name__ == "__main__":
    # 初始化模型
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'dice_loss': dice_loss}
    )
    print("模型加载成功，输入规格：", model.input_shape)

    # 获取测试文件列表
    mhd_files = [os.path.join(TEST_DIR, f)
                 for f in os.listdir(TEST_DIR) if f.endswith('.mhd')]
    print(f"发现 {len(mhd_files)} 个测试文件")

    # 创建进程池
    with Pool(processes=WORKERS) as pool:
        tasks = [(f, model) for f in mhd_files]
        results = list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks)))

    # 生成报告
    generate_report(results)

    # 可视化示例
    visualize_sample_results()

    print("\n批量测试完成！预测结果保存在：", OUTPUT_DIR)