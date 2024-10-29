import os

import matplotlib.pyplot as plt
import numpy as np

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)

class Classification(object):
    _defaults = {

        "model_path"    : 'logs/ep081-loss0.009-val_loss0.000.h5',
        "classes_path"  : 'model_data/cls_classes.txt',
        "input_shape"   : [224, 224],
        "backbone"      : 'vgg16',
        "alpha"         : 0.25
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        if self.backbone == "mobilenet":
            self.model = get_model_from_name[self.backbone](input_shape = [self.input_shape[0], self.input_shape[1], 3], classes = self.num_classes, alpha = self.alpha)
        else:
            self.model = get_model_from_name[self.backbone](input_shape = [self.input_shape[0], self.input_shape[1], 3], classes = self.num_classes)
        self.model.load_weights(self.model_path)
        print('{} model, and classes {} loaded.'.format(model_path, self.class_names))
    def detect_image(self, image):
        image = cvtColor(image)  # 颜色空间变换
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])  # 图片缩放为固定大小
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)  # 预处理输入图片数据
        preds = self.model.predict(image_data)[0]
        class_name = self.class_names[np.argmax(preds)]  # 基于类别概率进行分类
        probability = np.max(preds)  # 获取最大概率值
        return class_name, probability  # 返回类别名称和其概率
