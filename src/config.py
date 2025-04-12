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

        self.epoch = 3
        self.input_size = 512       # 单维度
        self.num_classes = 1
        self.batch_size = 5

        # self.anchors = [
        #     [(142, 110), (192, 243), (459, 401)],  # 大尺度
        #     [(36, 75), (76, 55), (72, 146)],  # 中尺度
        #     [(12, 16), (19, 36), (40, 28)]  # 小尺度
        # ]

        # YOLO格式配置
        self.img_size = (512, 512)  # 输入图像尺寸
        self.grid_sizes = [16, 32, 64]  # YOLO特征图尺寸

        self.anchors = [
            # ---- 16x16网格（大目标层）----
            # 锚框尺寸应基于步长（通常为步长的0.5~2倍）
            [(32, 32), (48, 48), (64, 64)],  # 覆盖16~32像素目标

            # ---- 32x32网格（中目标层）----
            # 步长 = 512 / 32 = 16
            [(16, 16), (24, 24), (32, 32)],  # 覆盖8~16像素目标

            # ---- 64x64网格（小目标层）----
            # 步长 = 512 / 64 = 8
            [(8, 8), (12, 12), (16, 16)]  # 覆盖4~8像素目标
        ]

        self.hu_window = (-1000, 400)
        self.slice_thickness = 1.0
        self.class_id = 0

        # 预处理参数
        self.cube_size = 32  # 截取立方体尺寸
        self.hu_range = (-1000, 400)  # HU值截断范围

        # 预处理输出配置
        # self.PREPROCESS = {
        #     "output_dir": "./data/processed",
        #     "image_size": (512, 512),
        #     "classes": ["nodule_1", "nodule_2", "nodule_3", "nodule_4", "nodule_5"],
        #     "annotation_format": "yolo"  # 支持yolo/coco格式
        # }

        # 模型配置
        self.MODEL = {
            "input_shape": (512, 512, 1),
            "anchors": self.anchors,  # 使用之前定义的锚框
            "num_classes": 1,
            "model_save_path": "D:\\PycharmProjects\\classify_img\\model",
        }


config = Config()