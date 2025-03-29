# main.py - 执行入口
from config import config
from preprocessor import *
from model_train import YoloTrainer
from src.YoloLabelGenerator import YOLOLabelGenerator
from src.dataset import YoloDataset


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
    # # 1. 数据预处理
    # preprocessor = LunaYoloPreprocessor(config)
    #
    # # 先加载标注文件
    # df_annotations = pd.read_csv(config.annotation_csv)
    # print(f"成功加载标注文件，共 {len(df_annotations)} 条记录")
    #
    # # 处理每个CT文件
    # for mhd_file in glob.glob(f"{config.raw_data_dir}/*.mhd"):
    #     preprocessor._process_patient(mhd_file, df_annotations)

    # # 加载标注数据
    # annotations_df = pd.read_csv(config.annotation_csv)
    # # 初始化预处理器
    # preprocessor = LunaYoloPreprocessor(config)
    #
    # # 遍历处理所有CT文件
    # for root, dirs, files in os.walk(config.raw_data_dir):
    #     for file in files:
    #         if file.endswith(".mhd"):
    #             mhd_path = os.path.join(root, file)
    #             # 执行预处理（自动生成图像和标签）
    #             preprocessor.process_patient(mhd_path, annotations_df)
    #
    # train_data = create_dataset(config, config.batch_size)

    # preprocessor = LunaYoloPreprocessor(config)
    #
    # # 加载结节标注数据
    # annotations = pd.read_csv(config.annotation_csv)
    # validate_annotations(config)
    #
    # # # 处理每个CT文件
    # for mhd_file in glob.glob(f"{config.raw_data_dir}/*.mhd"):
    #     preprocessor.process_nodule(mhd_file, df_annotations)
    #


    preprocessor = LunaYoloPreprocessor(config)

    # # 加载标注数据
    annotations = pd.read_csv(config.annotation_csv)

    # 处理每个CT文件
    for mhd_path in glob.glob(f"{config.raw_data_dir}/*.mhd"):
        # 加载CT数据
        ct_scan = load_mhd(mhd_path)
        patient_id = ct_scan['seriesuid']

        # 获取对应标注
        patient_annots = annotations[annotations['seriesuid'] == patient_id]

        # 处理每个结节
        for _, annot_row in patient_annots.iterrows():
            preprocessor.process_nodule(ct_scan, annot_row.to_dict(), patient_id)


    # # 2. 加载数据集
    dataset = YoloDataset(config)
    train_data = dataset.load_dataset(config.PREPROCESS["output_dir"])




    # 3. 训练模型
    trainer = YoloTrainer(config)
    trainer.train(train_data, train_data)