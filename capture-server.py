# -*- coding: UTF-8 -*-
import cv2
import time
import dlib
from threading import Timer
import numpy as np
from kafka import KafkaProducer
import json
import base64

srcHash = ''


def has_face(frame, time):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    if (len(rects) > 0):
        return True
    return False


def face_cut(frame, time):
    # img = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
    img = frame
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    # 获取图片的宽高
    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    # dlib检测
    for k, d in enumerate(dets):
        # 计算矩形大小
        # (x,y), (宽度width, 高度height)
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])

        # 计算矩形框大小
        height = d.bottom() - d.top()
        width = d.right() - d.left()

        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            if d.top() + i >= img_height:  # 防止越界
                continue
            for j in range(width):
                if d.left() + j >= img_width:  # 防止越界
                    continue
                img_blank[i][j] = img[d.top() + i][d.left() + j]
        img_blank = cv2.resize(img_blank, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imencode('.jpg', img_blank)[1].tofile(str(time) + str(k) + ".jpg")  # 正确方法


def capture(frame, time, sender):
    if (has_face(frame, time)):
        print u'face detector successfull'
        if (is_not_duplicate(frame)):
            res = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)
            print u'save image'
            # kafka send
            captureImg = base64.b64encode(cv2.imencode(".jpg", res)[1].tostring())
            try:
                #todo:save into s3
                import S3Utils
                #S3Utils.upload(captureImg,)
                #todo:send to kafka to save metadata into  image-resoure-db
                print 'send to kafka successfully'
            except Exception as e:
                print e
                print 'Kafka send error ,save in local file system'


def is_not_duplicate(target):
    target_hash = dHash(target)
    global srcHash
    if srcHash is '':
        srcHash = target_hash
        return True
    else:
        if (cmpHash(srcHash, target_hash) > 5):
            srcHash = target_hash;
            return True
    return False


def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def face_feature(frame, detecotr, shape_predictor, face_rec_model):
    # img = cv2.imread(frame,cv2.IMREAD_COLOR)
    img = frame
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    dets = detecotr(img, 1)
    face_features = []
    for index, face in enumerate(dets):
        print('face {};left:{}top{};right{};bootpm {}'.format(index, face.left(), face.top(), face.right(),
                                                              face.bottom()))
        shape = shape_predictor(img2, face)
        # if you want to show 68 det points you can upflod the comment
        for i, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
        cv2.namedWindow(str(index), cv2.WINDOW_AUTOSIZE)
        cv2.imshow(str(index), img)
        face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)
        face_features.append(face_descriptor)
        print(face_descriptor)
    return face_features


def compare_face_feature(src, target):
    diff = 0
    # for v1, v2 in data1, data2:
    # diff += (v1 - v2)**2
    for i in xrange(len(src)):
        diff += (src[i] - target[i]) ** 2
    diff = np.sqrt(diff)
    print diff
    return diff


if __name__ == '__main__':
    rtsp = "rtsp://192.168.31.8:554/0000000000200000000000000530003:0000000000140000000000000500003:192.168.31.8:420000/MainStream"
    cap = cv2.VideoCapture(rtsp)
    ret = cap.read()
    while ret:
        frame = cap.read()
        # frame is a tuple,so use frame[1]
        Timer(0.04, capture(frame[1], str(time.time())))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
