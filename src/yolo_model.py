import tensorflow as tf
from tensorflow.keras import Model, layers, initializers

class ConvBNMish(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, use_bias=False):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=use_bias,
                                  kernel_initializer=initializers.HeNormal())
        self.bn = layers.BatchNormalization()

    @staticmethod
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.mish(x)


class CSPResBlock(layers.Layer):
    def __init__(self, filters, num_blocks, shortcut=True):
        super().__init__()
        self.conv1 = ConvBNMish(filters, 1)
        self.conv2 = ConvBNMish(filters, 1)
        self.blocks = [ConvBNMish(filters // 2, 1) for _ in range(num_blocks)]
        self.conv3 = ConvBNMish(filters, 1)
        self.shortcut = shortcut

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        for block in self.blocks:
            x2 = block(x2)
        x = tf.concat([x1, x2], axis=-1)
        x = self.conv3(x)
        if self.shortcut:
            return x + inputs
        return x


class CSPDarknet53(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNMish(32, 3)
        self.conv2 = ConvBNMish(64, 3, strides=2)
        self.csp1 = CSPResBlock(64, 1)
        self.conv3 = ConvBNMish(128, 3, strides=2)
        self.csp2 = CSPResBlock(128, 2)
        self.conv4 = ConvBNMish(256, 3, strides=2)
        self.csp3 = CSPResBlock(256, 8)
        self.conv5 = ConvBNMish(512, 3, strides=2)
        self.csp4 = CSPResBlock(512, 8)
        self.conv6 = ConvBNMish(1024, 3, strides=2)
        self.csp5 = CSPResBlock(1024, 4)

    def call(self, inputs):
        x = self.conv1(inputs)
        # print("CSPDarknet53输入形状:", x.shape)
        x = self.conv2(x)
        x = self.csp1(x)
        x = self.conv3(x)
        x = self.csp2(x)
        x = self.conv4(x)
        route_small = self.csp3(x)
        x = self.conv5(route_small)
        route_medium = self.csp4(x)
        x = self.conv6(route_medium)
        route_large = self.csp5(x)
        return route_small, route_medium, route_large


class SPP(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNMish(512, 1)
        self.maxpool1 = layers.MaxPool2D(13, 1, padding='same')
        self.maxpool2 = layers.MaxPool2D(9, 1, padding='same')
        self.maxpool3 = layers.MaxPool2D(5, 1, padding='same')
        self.conv_out = ConvBNMish(1024, 3)

    def call(self, inputs):
        x = self.conv(inputs)
        p1 = self.maxpool1(x)
        p2 = self.maxpool2(x)
        p3 = self.maxpool3(x)
        x = tf.concat([p3, p2, p1, x], axis=-1)
        return self.conv_out(x)


class PANet(Model):
    def __init__(self):
        super().__init__()
        self.conv_large = ConvBNMish(512, 1)
        self.upsample_large = layers.UpSampling2D(2)
        self.conv_medium = ConvBNMish(256, 1)
        self.upsample_medium = layers.UpSampling2D(2)
        self.conv_up1 = ConvBNMish(256, 3)
        self.conv_up2 = ConvBNMish(128, 3)
        self.conv_down1 = ConvBNMish(256, 3, strides=2)
        self.conv_down2 = ConvBNMish(512, 3, strides=1)

    def call(self, features):
        route_small, route_medium, route_large = features

        # 上采样路径
        x = self.conv_large(route_large)
        x = self.upsample_large(x)
        x = tf.concat([x, route_medium], axis=-1)
        x_medium = self.conv_up1(x)

        x = self.upsample_medium(x_medium)
        x = tf.concat([x, route_small], axis=-1)
        x_small = self.conv_up2(x)

        # 下采样路径
        y = self.conv_down1(x_small)
        y = tf.concat([y, route_medium], axis=-1)
        y_medium = self.conv_down2(y)

        # 确保输出形状正确
        # print("PANet 输出形状：")
        # print(f"x_small: {x_small.shape}")
        # print(f"y_medium: {y_medium.shape}")
        # print(f"route_large: {route_large.shape}")

        return x_small, y_medium, route_large


class YOLOHead(layers.Layer):
    def __init__(self, filters, num_anchors, num_classes, name=None):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = ConvBNMish(filters, 3)
        self.conv2 = layers.Conv2D(
            num_anchors * (5 + num_classes), 1,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            kernel_regularizer='l2'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        # Reshape输出为 (batch, grid, grid, num_anchors, 5 + num_classes)
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.num_anchors, 5 + self.num_classes))
        return x


# class YOLOv4(Model):
#     def __init__(self, num_classes, anchors, input_size):
#         super().__init__()
#         self.input_layer = layers.Input(shape=(input_size, input_size, 1))  # 添加输入层
#         self.num_classes = num_classes
#         self.anchors = anchors
#         self.input_size = input_size
#
#         self.backbone = CSPDarknet53()
#         self.spp = SPP()
#         self.panet = PANet()
#
#         # 调整检测头参数匹配特征图通道
#         self.head_large = YOLOHead(512, len(anchors[0]), num_classes, name="large")  # 大尺度13x13
#         self.head_medium = YOLOHead(256, len(anchors[1]), num_classes, name="medium")  # 中尺度26x26
#         self.head_small = YOLOHead(128, len(anchors[2]), num_classes, name="small")  # 小尺度52x52
#
#
#     def call(self, inputs):
#         route_small, route_medium, route_large = self.backbone(inputs)
#         x = self.spp(route_large)
#         x_small, x_medium, x_large = self.panet((route_small, route_medium, x))
#
#         # 调整输出顺序为小->中->大
#         return {
#             "large": self.head_large(x_large),  # 13x13
#             "medium": self.head_medium(x_medium),  # 26x26
#             "small": self.head_small(x_small)  # 52x52
#         }

# 模型集成（字典方式返回）
class YOLOv4(Model):
    def __init__(self, num_classes, anchors, input_size):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors  # 确保是Python列表
        self.input_size = input_size

        self.backbone = CSPDarknet53()
        self.spp = SPP()
        self.panet = PANet()

        # 调整检测头参数匹配特征图通道
        self.head_large = YOLOHead(512, len(anchors[0]), num_classes, name="large")
        self.head_medium = YOLOHead(256, len(anchors[1]), num_classes, name="medium")
        self.head_small = YOLOHead(128, len(anchors[2]), num_classes, name="small")

    def call(self, inputs):
        route_small, route_medium, route_large = self.backbone(inputs)
        x = self.spp(route_large)
        x_small, x_medium, x_large = self.panet((route_small, route_medium, x))

        return {
            "large": self.head_large(x_large),  # 13x13
            "medium": self.head_medium(x_medium),  # 26x26
            "small": self.head_small(x_small)  # 52x52
        }

    # 新增序列化方法 =====================
    def get_config(self):
        # 返回构造参数
        return {
            "anchors": self.anchors,
            "num_classes": self.num_classes,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, anchors, num_classes, reduction="sum_over_batch_size",  # 新增父类参数处理
                 name='yolo_loss', lambda_coord = 30.0, lambda_noobj=0.5):
        super().__init__(reduction=reduction, name=name)  # 关键：显式传递父类参数
        self.anchors = anchors
        self.num_classes = num_classes

        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 负样本置信度损失权重

    def call(self, y_true, y_pred):
        # 验证输入形状 对于y_true最后一个维度有(x,y,w,h,conf,prob)    置信度表示是否存在目标,类别概率表示目标类别的预测该v了
        # print("y_true 形状:", y_true.shape)
        # print("y_pred 形状:", y_pred.shape)
        # 输入验证
        tf.debugging.assert_shapes([
            (y_true, (None, None, None, None, 5 + self.num_classes)),
            (y_pred, (None, None, None, None, 5 + self.num_classes))
        ])

        obj_mask = y_true[..., 4:5]  # (B,H,W,A,1)
        noobj_mask = 1.0 - obj_mask

        # 坐标损失（带数值稳定性）
        pred_xy = tf.sigmoid(y_pred[..., 0:2])
        pred_wh = tf.clip_by_value(tf.exp(y_pred[..., 2:4]), 1e-3, 1e3)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        coord_diff = tf.square(y_true[..., :4] - pred_box)
        coord_loss = tf.reduce_sum(obj_mask * coord_diff, [1, 2, 3, 4]) * self.lambda_coord

        # 置信度损失（带梯度保护）
        pred_conf = tf.clip_by_value(tf.sigmoid(y_pred[..., 4:5]), 1e-7, 1.0 - 1e-7)

        # 正样本损失
        bce_obj = tf.keras.losses.binary_crossentropy(obj_mask, pred_conf, axis=-1)
        bce_obj = tf.expand_dims(bce_obj, -1)  # 保持5D形状
        conf_loss_obj = bce_obj * obj_mask

        # 负样本损失
        bce_noobj = tf.keras.losses.binary_crossentropy(obj_mask, pred_conf, axis=-1)
        bce_noobj = tf.expand_dims(bce_noobj, -1)
        conf_loss_noobj = bce_noobj * noobj_mask * self.lambda_noobj

        conf_loss = tf.reduce_sum(conf_loss_obj + conf_loss_noobj, [1, 2, 3, 4])

        # 分类损失（带标签平滑）
        # 分类损失修正
        true_cls = y_true[..., 5:5 + self.num_classes]
        pred_cls = tf.clip_by_value(
            tf.sigmoid(y_pred[..., 5:5 + self.num_classes]),
            1e-7, 1.0 - 1e-7
        )

        # 步骤1：计算交叉熵并保持维度
        cls_bce = tf.keras.losses.binary_crossentropy(
            true_cls, pred_cls, axis=-1
        )  # 输出形状 (B,13,13,3)

        # 步骤2：扩展维度以匹配mask
        cls_bce = tf.expand_dims(cls_bce, axis=-1)  # 形状变为 (B,13,13,3,1)

        # 步骤3：应用物体掩码
        cls_loss = tf.reduce_sum(
            obj_mask * cls_bce,
            axis=[1, 2, 3, 4]  # 正确减少维度
        )

        return coord_loss + conf_loss + cls_loss


    # def call(self, y_true, y_pred):


        # 确保输入维度正确
        # assert y_pred.shape[-1] == 5 + self.num_classes
        # assert y_true.shape[-1] == 5 + self.num_classes

        # 之前的方法
        # pred_box = y_pred[..., :4]
        # true_box = y_true[..., :4]
        # coord_loss = tf.reduce_sum(obj_mask * tf.square(true_box - pred_box), axis=[1, 2, 3, 4])
        # print("coord_loss 值:", coord_loss)

        # 置信度损失（Sigmoid处理）
        # pred_conf = tf.sigmoid(y_pred[..., 4:5])
        # true_conf = y_true[..., 4:5]
        # conf_loss = tf.keras.losses.binary_crossentropy(true_conf, pred_conf)
        # conf_loss = tf.reduce_sum(conf_loss * tf.squeeze(obj_mask, axis=-1), axis=[1, 2, 3])

        # # 分类损失（Sigmoid处理）
        # pred_cls = tf.sigmoid(y_pred[..., 5:5 + self.num_classes])
        # true_cls = y_true[..., 5:5 + self.num_classes]
        # cls_loss = tf.keras.losses.binary_crossentropy(true_cls, pred_cls)
        # cls_loss = tf.reduce_sum(cls_loss * tf.squeeze(obj_mask, axis=-1), axis=[1, 2, 3])

        # # 总损失（保持batch维度）
        # total_loss = coord_loss + conf_loss + cls_loss
        # return total_loss

    def get_config(self):
        # 包含父类参数
        config = super().get_config()  # 获取父类配置
        config.update({
            "anchors": self.anchors,
            "num_classes": self.num_classes,
            # "lambda_coord": self.lambda_coord,
            # "lambda_noobj": self.lambda_noobj
        })
        return config




# YOLOv4模型集成(列表方式返回）
# class LunaYOLOv4(tf.keras.Model):
#     def __init__(self, config):
#         super().__init__()
#         # 子类化不需要显式定义输入层
#         self.backbone = CSPDarknet53()
#         self.neck = PANet()
#
#         # 检测头
#         self.head_large = YOLOHead(512, len(config.anchors[0]), config.num_classes)
#         self.head_medium = YOLOHead(256, len(config.anchors[1]), config.num_classes)
#         self.head_small = YOLOHead(128, len(config.anchors[2]), config.num_classes)
#
#         # 多尺度训练配置
#         self.grid_sizes = config.grid_sizes
#         self.anchors = config.anchors
#         self.output_names = ['large', 'medium', 'small']
#
#         self.loss_metrics = {
#             'total_loss': tf.keras.metrics.Mean(name='total_loss'),
#             'coord_loss': tf.keras.metrics.Mean(name='coord_loss'),
#             'conf_loss': tf.keras.metrics.Mean(name='conf_loss')
#         }
#
#     def call(self, inputs, training=False):
#         # 主干网络前向传播
#         route_small, route_medium, route_large = self.backbone(inputs)
#
#         # 特征金字塔融合
#         x_small, x_medium, x_large = self.neck(
#             (route_small, route_medium, route_large)
#         )
#
#         # 多尺度预测输出
#         outputs = [
#             self.head_large(x_large),       # (batch, 13, 13, 3, 5+num_classes)
#             self.head_medium(x_medium),     # (batch, 26, 26, 3, 5+num_classes)
#             self.head_small(x_small)        # (batch, 52, 52, 3, 5+num_classes)
#         ]
#         return outputs
