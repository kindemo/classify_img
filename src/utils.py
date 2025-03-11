import tensorflow as tf
class AdaptiveCombinedLoss(tf.keras.losses.Loss):
    def __init__(self, initial_alpha=0.5, gamma=2.0, smooth=1e-6, name='adaptive_combined_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.smooth = smooth
        # 可学习的alpha参数（初始化为偏向负样本）
        self.alpha = tf.Variable(initial_alpha, dtype=tf.float32, trainable=True)

    def dice_loss(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

    def focal_loss(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # 动态计算正样本比例
        positive_ratio = tf.reduce_mean(y_true)
        dynamic_alpha = self.alpha * positive_ratio

        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = -tf.reduce_mean(
            dynamic_alpha * (1 - pt)  ** self.gamma * tf.math.log(pt + self.smooth)
        )
        return focal

    def call(self, y_true, y_pred):
        dice = self.dice_loss(y_true, y_pred)
        focal = self.focal_loss(y_true, y_pred)

        # 动态调整权重（Dice主导，Focal辅助）
        total_loss = dice + 0.3 * focal
        return total_loss

    def get_config(self):
        return {"gamma": self.gamma, "smooth": self.smooth}



def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def combined_loss(alpha=0.5, gamma=2):
    def loss(y_true, y_pred):
        # Dice loss
        dice = dice_loss(y_true, y_pred)

        # Focal loss
        epsilon = 1e-5
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = -tf.reduce_mean(alpha * (1 - pt) ** gamma * tf.math.log(pt))

        return dice + focal

    return loss