
import os
import cv2
import numpy as np

from PIL import Image

from classification import Classification

# 实例化
classfication = Classification()

face_cascade = cv2.CascadeClassifier(r'model_data/haarcascade_frontalface_alt.xml')


def face_detect(img):

    # face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
    img = img

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
        # minSize=(120, 120)
    )
    return img, img_gray, faces


def generate_faces(face_img, img_size=48):

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img,0))
    # resized_images.append(cv2.flip(face_img[2], 1))
    # resized_images.append(cv2.flip(face_img[3], 1))
    # resized_images.append(cv2.flip(face_img[4], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression():
    border_color = (255, 0, 0)  # 框框颜色
    # font_color = (255, 255, 255)  # 白字字
    font_color = (0, 0, 255)  # 字字颜色
    # 255 20 147
    # 视频流输入来源
    #video_path = input(r"./test.mp4")
    capture = cv2.VideoCapture(r'.\test.avi')  # 视频检测
    #   save
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not os.path.exists("output"):
        os.makedirs("output")
    # out = cv2.VideoWriter('output/'+'save.avi', fourcc, 30, size)
    out = cv2.VideoWriter('output/' + 'save.avi', fourcc, fps, size)
    #

    while (True):
        _, frame = capture.read()

        # 格式转变，BGRtoRGB
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        # frame = Image.fromarray(np.uint8(frame))

        img, img_gray, faces = face_detect(frame)
        # cv2.imshow("img_gray", img_gray) 显示正常图像

        if len(faces) == 0:
            # print('no face detect.')
            cv2.putText(img, 'No Face Detect.', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 遍历每一个脸
        # emotions = []
        # result_possibilitys = []

        for (x, y, w, h) in faces:
            face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
            # 转换格式
            face_img_gray = Image.fromarray(face_img_gray)
            # cv2.imshow("face_img_gray", face_img_gray)
            # faces_img_gray = generate_faces(face_img_gray)
            # cv2.imshow("faces_img_gray", faces_img_gray)

            # 预测结果
            class_name = classfication.detect_image(face_img_gray)
            # class_name = 'test'

            # results = model.predict(faces_img_gray)
            # result_sum = np.sum(results, axis=0).reshape(-1)
            # label_index = np.argmax(result_sum, axis=0)
            # emotion = index2emotion(label_index, 'en')

            # print(class_name)
            emotion = class_name[0]
            probability = class_name[1]
            # print(emotion)

            # 画出人脸矩形
            cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
            # 显示情绪类别
            cv2.putText(img, emotion, (x + 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, font_color, 5)
            # 显示类别的加权概率结果
            cv2.putText(img, str(probability), (x + 30, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 1)

            # ['disgust', 'fear', 'happiness', 'others', 'repression', 'sadness', 'surprise']

        cv2.imshow("Video", img)
        # save video
        out.write(frame)

        key = cv2.waitKey(30)
        # 如果输入esc则退出循环
        if key == 27:
            break

    capture.release()  # 释放摄像头
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_expression()
