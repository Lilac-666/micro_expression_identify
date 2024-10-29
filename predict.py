from PIL import Image

from classification import Classification
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
classfication = Classification()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name, probability = classfication.detect_image(image)
        print(class_name,probability)
