# config.py - 配置文件
import os



class Config:
    def __init__(self):
        # 数据路径
        self.raw_data_dir = "D:\BaiduNetdiskDownload\LUNA16\subset0"
        self.processed_dir = "D:\BaiduNetdiskDownload\LUNA16\processed"
        self.annotation_csv = "D:/BaiduNetdiskDownload/LUNA16/CSVFILES/annotations.csv"

        self.output_excel_path = "D:\BaiduNetdiskDownload\LUNA16\CSVFILES\yolo_position.xlsx"

        self.image_dir = "D:\BaiduNetdiskDownload\LUNA16\processed\images"
        self.label_dir = "D:\BaiduNetdiskDownload\LUNA16\processed\labels"

        # 验证路径
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {self.raw_data_dir}")
        if not os.path.exists(self.annotation_csv):
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_csv}")

        self.epoch = 5
        self.input_size = 416       # 单维度
        self.num_classes = 1
        self.batch_size = 16

        self.spacing = (1.0, 1.0, 1.0)  # 添加空间间距参数

        # YOLO格式配置
        self.img_size = (416, 416)  # 输入图像尺寸
        self.grid_sizes = [52, 26, 13]  # YOLO特征图尺寸
        self.anchors = [
            [(142, 110), (192, 243), (459, 401)],  # 大尺度
            [(36, 75), (76, 55), (72, 146)],  # 中尺度
            [(12, 16), (19, 36), (40, 28)]  # 小尺度
        ]

        self.hu_window = (-1000, 400)
        self.slice_thickness = 1.0
        self.class_id = 1

        # 预处理参数
        self.target_spacing = 1.0  # 体素标准化间距(mm)
        self.cube_size = 32  # 截取立方体尺寸
        self.hu_range = (-1000, 400)  # HU值截断范围

        # 预处理输出配置
        self.PREPROCESS = {
            "output_dir": "./data/processed",
            "image_size": (416, 416),
            "classes": ["nodule_1", "nodule_2", "nodule_3", "nodule_4", "nodule_5"],
            "annotation_format": "yolo"  # 支持yolo/coco格式
        }

        # 模型配置
        self.MODEL = {
            "input_shape": (416, 416, 1),
            "anchors": self.anchors,  # 使用之前定义的锚框
            "num_classes": 1,
            "model_save_path": "D:\\PycharmProjects\\classify_img\\model",
        }


config = Config()