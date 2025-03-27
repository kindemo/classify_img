from yolo_model import YOLOv4Loss, YOLOv4


# model.py - 模型接口
def create_yolov4(config):
    model = YOLOv4(
        num_classes=config.MODEL["num_classes"],
        anchors=config.MODEL["anchors"],
        input_size=config.MODEL["input_shape"]
    )
    return model


class YoloTrainer:
    def __init__(self, config):
        self.model = create_yolov4(config)
        self.loss_fn = YOLOv4Loss(
            anchors=config.MODEL["anchors"],
            num_classes=config.MODEL["num_classes"],
            input_size=config.MODEL["input_shape"][0]
        )

    def train(self, train_dataset, val_dataset):
        self.model.compile(optimizer="adam", loss=self.loss_fn)
        self.model.fit(train_dataset, validation_data=val_dataset)