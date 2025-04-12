from keras.callbacks import ModelCheckpoint

from src.config import config
from yolo_model import YoloLoss, YOLOv4
import tensorflow as tf



class YoloTrainer:
    def __init__(self, config):
        # 初始化模型（需确保输入尺寸匹配）
        self.model = YOLOv4(
            num_classes=config.num_classes,
            anchors=config.MODEL["anchors"],  # 确保格式为[[large_anchors], [medium_anchors], [small_anchors]]
            input_size=config.input_size  # 例如 416
        )

        self.loss_fn  = {
            'large': YoloLoss(config.anchors[0], config.num_classes, reduction="sum_over_batch_size"),  # large层损失
            'medium': YoloLoss(config.anchors[1], config.num_classes, reduction="sum_over_batch_size"),  # medium层损失
            'small': YoloLoss(config.anchors[2], config.num_classes, reduction="sum_over_batch_size")  # small层损失
        }

        # 创建回调函数以保存模型
        self.checkpoint_callback = ModelCheckpoint(
            filepath=config.MODEL["model_save_path"],
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

    def train(self, train_dataset, val_dataset):
        self.model.compile(optimizer="adam",loss=self.loss_fn)

        # # 编译时设置loss_weights字典
        # self.model.compile(
        #     optimizer="adam",
        #     loss=self.loss_fn,
        #     loss_weights={
        #         'large': 0.1,
        #         'medium': 0.3,
        #         'small': 0.6
        #     }
        # )


        self.model.fit(train_dataset,
                       callbacks=[self.checkpoint_callback],
                       validation_data=val_dataset,  # 添加验证数据
                       epochs=config.epoch,
                       batch_size=config.batch_size)





# if __name__ == '__main__':
#     # 添加模型结构检查代码
#     def verify_model_structure():
#         test_input = tf.random.normal((1, 416, 416, 1))
#
#         # 测试前向传播
#         outputs = LunaYOLOv4(test_input)
#         print("Output shapes验证:")
#         print(f"Large scale: {outputs[0].shape}")  # 应输出(1,13,13,3,6)
#         print(f"Medium scale: {outputs[1].shape}")  # 应输出(1,26,26,3,6)
#         print(f"Small scale: {outputs[2].shape}")  # 应输出(1,52,52,3,6)
#
#         # 打印模型摘要
#         model.keras_model.summary()
#
#
#     verify_model_structure()



# if __name__ == '__main__':
#     # 测试模型前向传播
#     model = LunaYOLOv4(config)
#     test_input = tf.random.normal((1, 416, 416, 1))  # 单通道输入
#     outputs = model(test_input)
#     print("Output Shapes:")
#     print(f"Large: {outputs[0].shape}")  # 应输出 (1,13,13,3,6)
#     print(f"Medium: {outputs[1].shape}")  # 应输出 (1,26,26,3,6)
#     print(f"Small: {outputs[2].shape}")  # 应输出 (1,52,52,3,6)