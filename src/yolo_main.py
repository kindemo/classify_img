# main.py - 执行入口
from config import config
from preprocessor import *
from model_train import YoloTrainer
from src.YoloLabelGenerator import YOLOLabelGenerator



def load_mhd(mhd_path):
    """加载MHD文件，返回数据数组、原点、间距（保持原始x,y,z顺序）"""
    image = sitk.ReadImage(mhd_path)
    data = sitk.GetArrayFromImage(image)  # 形状为(z,y,x)
    origin = np.array(image.GetOrigin())  # x,y,z顺序
    spacing = np.array(image.GetSpacing())  # x,y,z顺序
    return {
        'data': data,
        'origin': origin,
        'spacing': spacing,
        'seriesuid': Path(mhd_path).stem
    }



if __name__ == "__main__":

    ## 1.数据预处理
    # preprocessor = LunaYoloPreprocessor(config)
    #
    # # # 加载标注数据
    # annotations = pd.read_csv(config.annotation_csv)
    #
    # # 处理每个CT文件
    # for mhd_path in glob.glob(f"{config.raw_data_dir}/*.mhd"):
    #     # 加载CT数据
    #     ct_scan = load_mhd(mhd_path)
    #     patient_id = ct_scan['seriesuid']
    #
    #     # 获取对应标注
    #     patient_annots = annotations[annotations['seriesuid'] == patient_id]
    #
    #     # 处理每个结节
    #     for _, annot_row in patient_annots.iterrows():
    #         preprocessor.process_nodule(ct_scan, annot_row.to_dict(), patient_id)


    # # 2. 加载数据集
    # dataset = YoloDataset(config)
    # train_data = dataset.load_dataset(config.PREPROCESS["output_dir"])

    train_data = create_dataset(config, config.batch_size)



    # 3. 训练模型
    trainer = YoloTrainer(config)
    trainer.train(train_data, train_data)