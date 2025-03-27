import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np
import cv2

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
        print("PANet 输出形状：")
        print(f"x_small: {x_small.shape}")
        print(f"y_medium: {y_medium.shape}")
        print(f"route_large: {route_large.shape}")

        return x_small, y_medium, route_large

class YOLOHead(layers.Layer):
    def __init__(self, filters, num_anchors, num_classes):
        super().__init__()
        self.conv1 = ConvBNMish(filters, 3)
        self.conv2 = layers.Conv2D(num_anchors * (5 + num_classes), 1,
                                   kernel_initializer=initializers.HeNormal())

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)


class YOLOv4(Model):
    def __init__(self, num_classes, anchors, input_size):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size

        self.backbone = CSPDarknet53()
        self.spp = SPP()
        self.panet = PANet()

        # 调整检测头参数匹配特征图通道
        self.head_large = YOLOHead(512, len(anchors[0]), num_classes)  # 大尺度13x13
        self.head_medium = YOLOHead(256, len(anchors[1]), num_classes)  # 中尺度26x26
        self.head_small = YOLOHead(128, len(anchors[2]), num_classes)  # 小尺度52x52

    def call(self, inputs):
        route_small, route_medium, route_large = self.backbone(inputs)
        x = self.spp(route_large)
        x_small, x_medium, x_large = self.panet((route_small, route_medium, x))

        # 调整输出顺序为小->中->大
        return (
            self.head_small(x_small),  # 52x52
            self.head_medium(x_medium),  # 26x26
            self.head_large(x_large)  # 13x13
        )


class YOLOv4Loss(tf.keras.losses.Loss):
    def __init__(self, anchors, num_classes, input_size, label_smoothing=0.1):
        super().__init__()
        self.anchors = np.array(anchors)
        self.num_classes = num_classes
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size[:2]
        self.label_smoothing = label_smoothing
        self.epsilon = 1e-9

    def _process_predictions(self, pred, anchors, scale):
        grid_size = tf.shape(pred)[1]
        pred = tf.reshape(pred, [-1, grid_size, grid_size, 3, 5 + self.num_classes])
        box_xy = tf.sigmoid(pred[..., 0:2])
        box_wh = tf.exp(pred[..., 2:4]) * anchors / self.input_size[0]
        conf = tf.sigmoid(pred[..., 4:5])
        prob = tf.sigmoid(pred[..., 5:])
        return box_xy, box_wh, conf, prob

    def _ciou_loss(self, boxes1, boxes2):
        b1_xy, b1_wh = boxes1[..., 0:2], boxes1[..., 2:4]
        b1_min = b1_xy - b1_wh / 2
        b1_max = b1_xy + b1_wh / 2
        b2_xy, b2_wh = boxes2[..., 0:2], boxes2[..., 2:4]
        b2_min = b2_xy - b2_wh / 2
        b2_max = b2_xy + b2_wh / 2

        intersect_min = tf.maximum(b1_min, b2_min)
        intersect_max = tf.minimum(b1_max, b2_max)
        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area

        iou = intersect_area / (union_area + self.epsilon)
        center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)

        enclose_min = tf.minimum(b1_min, b2_min)
        enclose_max = tf.maximum(b1_max, b2_max)
        enclose_wh = enclose_max - enclose_min
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        v = (4 / (np.pi ** 2)) * tf.square(
            tf.math.atan(b1_wh[..., 0] / (b1_wh[..., 1] + self.epsilon)) -
            tf.math.atan(b2_wh[..., 0] / (b2_wh[..., 1] + self.epsilon))
        )
        alpha = v / (1 - iou + v + self.epsilon)
        ciou = iou - (center_distance / (enclose_diagonal + self.epsilon) + alpha * v)
        return 1.0 - ciou

    def call(self, y_true, y_pred):
        total_loss = 0.0
        for scale in range(3):
            pred = y_pred[scale]
            true = y_true[scale]

            grid_size = tf.shape(pred)[1]
            tf.debugging.assert_equal(
                tf.shape(true)[1],
                grid_size,
                message=f"Scale {scale} grid_size mismatch"
            )

            # 处理预测值
            anchors = self.anchors[scale]
            box_xy, box_wh, conf, prob = self._process_predictions(pred, anchors, scale)

            # 处理真实值
            true_box_xy = true[..., 0:2]
            true_box_wh = true[..., 2:4]
            true_conf = true[..., 4:5]
            true_cls = true[..., 5:]  # 注意这里不需要添加批次维度

            # 确保true_cls和prob形状匹配
            true_cls = tf.expand_dims(true_cls, axis=0)  # 添加类别维度

            # 计算分类损失
            prob_loss = tf.keras.losses.binary_crossentropy(
                true_cls,
                prob,
                from_logits=False,
                label_smoothing=self.label_smoothing
            )

            # 应用对象掩码
            obj_mask = tf.squeeze(true_conf, axis=-1)  # 去除最后一个维度
            prob_loss = tf.reduce_sum(prob_loss * obj_mask) / (tf.reduce_sum(obj_mask) + self.epsilon)

            total_loss += prob_loss
        return total_loss






class NoduleDetector:
    def __init__(self, model_path, anchors, input_size=(416, 416), conf_thresh=0.5, iou_thresh=0.4):
        # 加载模型时注册所有自定义对象
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'YOLOv4Loss': YOLOv4Loss}
        )
        self.anchors = np.array(anchors)  # 转换为numpy数组
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        # 定义不同尺度对应的锚框索引
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 对应大、中、小三个尺度

    def non_max_suppression(self, boxes, scores, classes):
        """改进后的NMS实现，支持多类别"""
        selected_indices = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=100,
            iou_threshold=self.iou_thresh,
            score_threshold=self.conf_thresh
        )
        selected_boxes = tf.gather(boxes, selected_indices)
        selected_scores = tf.gather(scores, selected_indices)
        selected_classes = tf.gather(classes, selected_indices)
        return selected_boxes, selected_scores, selected_classes

    def decode_scale_pred(self, pred, scale_idx):
        """解码单个尺度的预测结果"""
        anchors = self.anchors[self.anchor_masks[scale_idx]]
        grid_size = pred.shape[1]  # 特征图尺寸 (h, w)
        pred = tf.reshape(pred, [1, grid_size, grid_size, len(anchors), 5 + 1])  # 假设num_classes=1

        # 分解预测分量
        box_xy = tf.sigmoid(pred[..., 0:2])
        box_wh = tf.exp(pred[..., 2:4]) * anchors / self.input_size[0]
        conf = tf.sigmoid(pred[..., 4:5])
        prob = tf.sigmoid(pred[..., 5:6])  # 假设单分类

        # 生成网格坐标
        grid_y, grid_x = tf.meshgrid(
            tf.range(grid_size, dtype=tf.float32),
            tf.range(grid_size, dtype=tf.float32)
        )
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(grid, 2)  # (h, w, 1, 2)

        # 转换到全局坐标
        box_xy = (box_xy + grid) / grid_size
        boxes = tf.concat([box_xy, box_wh], axis=-1)  # [x_center, y_center, width, height]

        # 转换为角点坐标
        boxes_corners = tf.concat([
            boxes[..., 0:2] - boxes[..., 2:4] / 2,  # xmin, ymin
            boxes[..., 0:2] + boxes[..., 2:4] / 2  # xmax, ymax
        ], axis=-1)

        # 展平结果
        boxes_corners = tf.reshape(boxes_corners, [-1, 4])
        conf = tf.reshape(conf, [-1])
        prob = tf.reshape(prob, [-1])
        scores = conf * prob  # 综合置信度

        # 过滤低置信度
        valid_mask = scores > self.conf_thresh
        boxes_corners = tf.boolean_mask(boxes_corners, valid_mask)
        scores = tf.boolean_mask(scores, valid_mask)
        classes = tf.ones_like(scores, dtype=tf.int32)  # 单分类问题

        return boxes_corners, scores, classes

    def decode_predictions(self, preds):
        """合并三个尺度的预测结果"""
        all_boxes = []
        all_scores = []
        all_classes = []

        for scale in range(3):
            boxes, scores, classes = self.decode_scale_pred(preds[scale], scale)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        # 合并所有预测
        boxes = tf.concat(all_boxes, axis=0)
        scores = tf.concat(all_scores, axis=0)
        classes = tf.concat(all_classes, axis=0)

        # 执行NMS
        if boxes.shape[0] > 0:
            final_boxes, final_scores, final_classes = self.non_max_suppression(
                boxes, scores, classes)
        else:
            return np.array([]), np.array([]), np.array([])

        return final_boxes.numpy(), final_scores.numpy(), final_classes.numpy()

    def preprocess_image(self, image):
        """改进的预处理流程，保持比例的缩放和填充"""
        # 转换为单通道
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 获取原始尺寸
        h, w = image.shape[:2]

        # 计算缩放比例
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 保持比例的缩放
        resized = cv2.resize(image, (new_w, new_h))

        # 创建填充图像
        padded = np.zeros(self.input_size, dtype=np.float32)
        dh = (self.input_size[0] - new_h) // 2
        dw = (self.input_size[1] - new_w) // 2
        padded[dh:dh + new_h, dw:dw + new_w] = resized

        # 添加批次和通道维度
        return np.expand_dims(padded, axis=(0, -1)) / 255.0

    def detect(self, ct_volume):
        """改进后的检测流程"""
        detections = []

        for slice_idx in range(ct_volume.shape[0]):
            # 获取CT切片
            slice_img = ct_volume[slice_idx].astype(np.float32)

            try:
                # 预处理
                processed = self.preprocess_image(slice_img)

                # 模型预测
                preds = self.model(processed)

                # 解码预测结果
                boxes, scores, classes = self.decode_predictions(preds)

                # 转换回原始坐标
                if len(boxes) > 0:
                    # 去除填充
                    scale = min(self.input_size[0] / slice_img.shape[0],
                                self.input_size[1] / slice_img.shape[1])
                    new_h = int(slice_img.shape[0] * scale)
                    new_w = int(slice_img.shape[1] * scale)
                    dh = (self.input_size[0] - new_h) // 2
                    dw = (self.input_size[1] - new_w) // 2

                    # 调整坐标到原始尺寸
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * self.input_size[1] - dw) / scale
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * self.input_size[0] - dh) / scale

                    # 限制边界
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, slice_img.shape[1])
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, slice_img.shape[0])

            except Exception as e:
                print(f"Error processing slice {slice_idx}: {str(e)}")
                boxes = np.array([])
                scores = np.array([])
                classes = np.array([])

            detections.append({
                "slice": slice_idx,
                "boxes": boxes,
                "scores": scores,
                "classes": classes
            })

        return detections


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = NoduleDetector("best_model.h5", ANCHORS)

    # 加载CT序列（示例数据）
    sample_ct = np.load("sample_ct.npy")  # shape: (depth, height, width)

    # 执行检测
    results = detector.detect(sample_ct)

    # 可视化结果
    for result in results:
        slice_idx = result["slice"]
        boxes = result["boxes"]
        print(f"Slice {slice_idx} detected {len(boxes)} nodules")