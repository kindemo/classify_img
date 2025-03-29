from keras.callbacks import ModelCheckpoint

from src.config import config
from yolo_model import YoloLoss
import tensorflow as tf


class LunaYOLOv4(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        # 显式定义输入层
        self.input_layer = tf.keras.layers.Input(shape=(config.input_size, config.input_size, 3), name='input_image')


class YoloTrainer:
    def __init__(self, config):
        # self.model = create_yolov4(config)
        self.model = LunaYOLOv4(config)

        self.loss_fn  = {
            'large': YoloLoss(config),  # large层损失
            'medium': YoloLoss(config),  # medium层损失
            'small': YoloLoss(config)  # small层损失
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
        self.model.fit(train_dataset, callbacks=[self.checkpoint_callback], epochs=config.epoch, batch_size=config.batch_size)