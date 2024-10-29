import numpy as np
from PIL import Image

# 从classification.py中导入Classification、cvtColor、preprocess_input
from classification import Classification, cvtColor, preprocess_input

# 从utils/utils.py中导入letterbox_image函数
from utils.utils import letterbox_image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 继承自Classification的top1_Classification类
class top1_Classification(Classification):

    # 对图像进行分类
    def detect_image(self, image):

        # 将图像转换为RGB格式
        image = cvtColor(image)

        # 调整图像大小
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])

        # 将图像数据转换为numpy数组
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        # 对图像进行预测
        preds = self.model.predict(image_data)[0]

        # 获取预测结果中最大值的下标
        arg_pred = np.argmax(preds)

        # 返回最大值的下标
        return arg_pred

# 评估模型的top-1精度
def evalute(classfication, lines):
    correct = 0
    y_pred = []
    y_true = []
    # 样本总数
    total = len(lines)
    # 逐行读取样本
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        y_true.append(y)
        y_pred.append(pred)

        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    return y_true,y_pred


classfication = top1_Classification()
with open("./cls_test.txt","r") as f:
    lines = f.readlines()
y_t,y_p = evalute(classfication, lines)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_t,y_p ))
