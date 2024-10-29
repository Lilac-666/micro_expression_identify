import numpy as np
from PIL import Image

from classification import Classification, cvtColor, preprocess_input
from utils.utils import letterbox_image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class top1_Classification(Classification):
    def detect_image(self, image):
        image       = cvtColor(image)
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        preds = self.model.predict(image_data)[0]
        arg_pred = np.argmax(preds)
        return arg_pred

def evaluteTop1(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += pred == y
        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    return correct / total

classfication = top1_Classification()
with open("./cls_test.txt","r") as f:
    lines = f.readlines()
top1 = evaluteTop1(classfication, lines)
print("top-1 accuracy = %.2f%%" % (top1*100))
