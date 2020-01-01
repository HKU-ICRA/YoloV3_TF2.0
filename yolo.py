"""
The following builds are used as reference:
    1. https://github.com/wizyoung/YOLOv3_TensorFlow
"""

import numpy as np
import tensorflow as tf


class Residual_block(tf.keras.Model):

    def __init__(self, nfilters1, nfilters2):
        super(Residual_block, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=nfilters1, kernel_size=1, strides=1, padding='same')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=nfilters2, kernel_size=3, strides=1, padding='same')
    
    def call(self, inputs):
        shortcut = inputs
        output = self.conv2d_1(inputs)
        output = self.conv2d_2(output)
        return output + shortcut


class DarkNet53(tf.keras.Model):

    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.residualBlock_3 = [Residual_block(32, 64)]
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
        self.residualBlock_5 = [Residual_block(64, 128) for _ in range(2)]
        self.conv2d_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')
        self.residualBlock_7 = [Residual_block(128, 256) for _ in range(8)]
        self.conv2d_8 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')
        self.residualBlock_9 = [Residual_block(256, 512) for _ in range(8)]
        self.conv2d_10 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')
        self.residualBlock_11 = [Residual_block(512, 1024) for _ in range(4)]
    
    def call(self, inputs):
        output = self.conv2d_1(inputs)
        output = self.conv2d_2(output)
        for residualBlock in self.residualBlock_3:
            output = residualBlock(output)
        output = self.conv2d_4(output)
        for residualBlock in self.residualBlock_5:
            output = residualBlock(output)
        output = self.conv2d_6(output)
        for residualBlock in self.residualBlock_7:
            output = residualBlock(output)
        route1 = output
        output = self.conv2d_8(output)
        for residualBlock in self.residualBlock_9:
            output = residualBlock(output)
        route2 = output
        output = self.conv2d_10(output)
        for residualBlock in self.residualBlock_11:
            output = residualBlock(output)
        return route1, route2, output


class YoloBlock(tf.keras.Model):

    def __init__(self, nfilters1, nfilters2):
        super(YoloBlock, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=nfilters1, kernel_size=1, strides=1, padding='same')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=nfilters2, kernel_size=3, strides=1, padding='same')
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=nfilters1, kernel_size=1, strides=1, padding='same')
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=nfilters2, kernel_size=3, strides=1, padding='same')
        self.conv2d_5 = tf.keras.layers.Conv2D(filters=nfilters1, kernel_size=1, strides=1, padding='same')
        self.conv2d_6 = tf.keras.layers.Conv2D(filters=nfilters2, kernel_size=3, strides=1, padding='same')
    
    def call(self, inputs):
        output = self.conv2d_1(inputs)
        output = self.conv2d_2(output)
        output = self.conv2d_3(output)
        output = self.conv2d_4(output)
        output = self.conv2d_5(output)
        route = output
        output = self.conv2d_6(output)
        return route, output


class Network(tf.keras.Model):

    def __init__(self, nclasses, img_size, anchors):
        super(Network, self).__init__()
        self.nclasses = nclasses
        self.img_size = img_size
        self.anchors = anchors
        self.darknet_1 = DarkNet53()
        self.yolo_2 = YoloBlock(512, 1024)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=3 * (5 + nclasses), kernel_size=1, strides=1, padding='same', bias_initializer=tf.zeros_initializer)
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.upsample_5 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.yolo_6 = YoloBlock(256, 512)
        self.conv2d_7 = tf.keras.layers.Conv2D(filters=3 * (5 + nclasses), kernel_size=1, strides=1, padding='same', bias_initializer=tf.zeros_initializer)
        self.conv2d_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same')
        self.upsample_9 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.yolo_10 = YoloBlock(128, 256)
        self.conv2d_11 = tf.keras.layers.Conv2D(filters=3 * (5 + nclasses), kernel_size=1, strides=1, padding='same', bias_initializer=tf.zeros_initializer)

    def bound_layer(self, f_map, anchors):
        gridDim = tf.shape(f_map)[1:3]
        f_map = tf.reshape(f_map, [-1, gridDim[0], gridDim[1], 3, 5 + self.nclasses])
        box_centers, box_sizes, conf_logits, class_logits = tf.split(f_map, [2, 2, 1, self.nclasses], axis=-1)
        box_centers = tf.sigmoid(box_centers)
        gridX = tf.range(gridDim[0], dtype=tf.int32)
        gridY = tf.range(gridDim[1], dtype=tf.int32)
        gridX, gridY = tf.meshgrid(gridX, gridY)
        x_offset = tf.reshape(gridX, (-1, 1))
        y_offset = tf.reshape(gridY, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.dtypes.cast(tf.reshape(x_y_offset, [gridDim[0], gridDim[1], 1, 2]), tf.float32)
        box_centers = box_centers + x_y_offset
        ratio = tf.dtypes.cast(self.img_size / gridDim, tf.float32)
        box_centers = box_centers * ratio[::-1]
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        box_sizes = box_sizes * ratio[::-1]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, class_logits

    def call(self, inputs):
        # Forward pass
        route1, route2, route3 = self.darknet_1(inputs)
        inter1, output = self.yolo_2(route3)
        fmap_1 = self.conv2d_3(output)
        inter1 = self.conv2d_4(inter1)
        inter1 = self.upsample_5(inter1)
        concat1 = tf.concat([inter1, route2], axis=3)
        inter2, output = self.yolo_6(concat1)
        fmap_2 = self.conv2d_7(output)
        inter2 = self.conv2d_8(inter2)
        inter2 = self.upsample_9(inter2)
        concat2 = tf.concat([inter2, route1], axis=3)
        _, output = self.yolo_10(concat2)
        fmap_3 = self.conv2d_11(output)
        fmap_anchors = [(fmap_1, self.anchors[6:9]), (fmap_2, self.anchors[3:6]), (fmap_3, self.anchors[0:3])]
        bounds = [self.bound_layer(fmap, anchor) for (fmap, anchor) in fmap_anchors]
        boxes_list, confs_list, classes_list = [], [], []
        for bound in bounds:
            x_y_offset, boxes, conf_logits, class_logits = bound
            gridDim = tf.shape(x_y_offset)[0:2]
            boxes = tf.reshape(boxes, [-1, gridDim[0] * gridDim[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, gridDim[0] * gridDim[1] * 3, 1])
            class_logits = tf.reshape(class_logits, [-1, gridDim[0] * gridDim[1] * 3, self.nclasses])
            confs = tf.sigmoid(conf_logits)
            classes = tf.sigmoid(class_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            classes_list.append(classes)
        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        classes = tf.concat(classes_list, axis=1)
        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
        return boxes, confs, classes
