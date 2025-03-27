# main.py - 执行入口
from config import config
from preprocessor import LunaPreprocessor
from dataset import YoloDataset
from model import YoloTrainer

if __name__ == "__main__":
    # 1. 数据预处理
    preprocessor = LunaPreprocessor(config)
    for src_path in data_sources:
        preprocessor.process_patient(src_path, patient_id)

    # 2. 加载数据集
    dataset = YoloDataset(config)
    train_data = dataset.load_dataset(config.PREPROCESS["output_dir"])

    # 3. 训练模型
    trainer = YoloTrainer(config)
    trainer.train(train_data)