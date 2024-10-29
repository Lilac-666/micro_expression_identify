import numpy as np
from PIL import Image

from classification import Classification, preprocess_input, cvtColor
from utils.utils import letterbox_image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class top5_Classification(Classification):
    def detect_image(self, image):
        image       = cvtColor(image)
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        preds = self.model.predict(image_data)[0]
        arg_pred = np.argsort(preds)[::-1]
        arg_pred_top5 = arg_pred[:5]
        return arg_pred_top5

def evaluteTop5(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += y in pred
        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    return correct / total

classfication = top5_Classification()
with open("./cls_test.txt","r") as f:
    lines = f.readlines()
top5 = evaluteTop5(classfication, lines)
print("top-5 accuracy = %.2f%%" % (top5*100))

