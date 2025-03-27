import tensorflow as tf
import numpy as np
import cv2
import os



class YoloDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_dir, anchors, num_classes, input_shape=(416, 416),
                 batch_size=8, shuffle=True, augment=True):
        """
        完整实现的YOLO数据生成器
        :param image_dir: CT切片图像目录
        :param label_dir: YOLO标注文件目录
        :param anchors: YOLO锚框列表
        :param num_classes: 类别数量
        :param input_shape: 网络输入尺寸 (h, w)
        :param batch_size: 批大小
        :param shuffle: 是否随机打乱数据
        :param augment: 是否应用数据增强
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.anchors = np.array(anchors)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_files = [f.replace('.png', '.txt') for f in self.image_files]
        self.indexes = np.arange(len(self.image_files))

        # 数据增强处理器
        self.augmentor = CTDataAugment() if augment else None

        # 初始化锚框
        self.anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])  # 对应3种尺度
        self.grid_sizes = [input_shape[0] // 32, input_shape[0] // 16, input_shape[0] // 8]

        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_images = []
        batch_labels = [np.zeros((self.batch_size, g, g, 3, 5 + self.num_classes))
                        for g in self.grid_sizes]

        for i, idx in enumerate(batch_indexes):
            # 加载图像
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 转为3通道

            # 加载标注
            label_path = os.path.join(self.label_dir, self.label_files[idx])
            boxes = self._parse_label(label_path)

            # 数据增强
            if self.augment:
                image, boxes = self.augmentor(image, boxes)

            # 预处理
            image, boxes = self._preprocess_data(image, boxes)
            batch_images.append(image)

            # 生成YOLO格式标签
            self._encode_labels(batch_labels, boxes, i)

        return np.array(batch_images) / 255.0, [l.astype(np.float32) for l in batch_labels]

    def _parse_label(self, label_path):
        """解析YOLO格式标注文件"""
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append([x_center, y_center, width, height, class_id])
        return np.array(boxes)

    def _preprocess_data(self, image, boxes):
        """图像预处理"""
        # 调整尺寸
        h, w = self.input_shape
        image = cv2.resize(image, (w, h))

        # 标准化坐标
        boxes[:, [0, 2]] *= w / image.shape[1]  # 宽度比例调整
        boxes[:, [1, 3]] *= h / image.shape[0]  # 高度比例调整

        return image, boxes

    def _encode_labels(self, batch_labels, boxes, batch_index):
        """生成YOLO训练所需的编码标签"""
        for box in boxes:
            x_center, y_center, width, height, class_id = box

            # 计算与锚框的最佳匹配
            best_iou = 0
            best_anchor = 0
            for i, anchor in enumerate(self.anchors):
                anchor_w, anchor_h = anchor
                iou = self._bbox_iou((0, 0, width, height), (0, 0, anchor_w, anchor_h))
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i

            # 确定特征图层级
            grid_scale = None
            for g, masks in enumerate(self.anchor_masks):
                if best_anchor in masks:
                    grid_scale = self.grid_sizes[g]
                    anchor_idx = np.where(masks == best_anchor)[0][0]
                    break

            if grid_scale is None:
                continue  # 没有匹配的锚框

            # 计算网格位置
            grid_x = int(x_center * grid_scale / self.input_shape[1])
            grid_y = int(y_center * grid_scale / self.input_shape[0])

            # 编码坐标
            x = x_center * grid_scale / self.input_shape[1] - grid_x
            y = y_center * grid_scale / self.input_shape[0] - grid_y
            w = np.log(width / self.anchors[best_anchor][0] + 1e-16)
            h = np.log(height / self.anchors[best_anchor][1] + 1e-16)

            # 填充标签数据
            batch_labels[g][batch_index, grid_y, grid_x, anchor_idx, 0:4] = [x, y, w, h]
            batch_labels[g][batch_index, grid_y, grid_x, anchor_idx, 4] = 1.0  # 置信度
            batch_labels[g][batch_index, grid_y, grid_x, anchor_idx, 5 + class_id] = 1.0  # 类别

    def _bbox_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (b1_area + b2_area - inter_area + 1e-16)

    def on_epoch_end(self):
        """每个epoch结束时打乱数据"""
        if self.shuffle:
            np.random.shuffle(self.indexes)