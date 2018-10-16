import cv2
import time
import dlib
from threading import Timer
import numpy as np
from kafka import KafkaProducer
import json
import base64

srcHash = ''


def has_face(frame, ):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rects = detector(gray, 1)
    if (len(rects) > 0):
        return True
    return False


def capture(frame, time, sender):
    if (has_face(frame)):
        print u'face detector successfull'
        if (is_not_duplicate(frame)):
            res = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)
            print u'save image'
            # kafka send
            captureImg = base64.b64encode(cv2.imencode(".jpg", res)[1].tostring())
            messgae = {"device": {"address": "position", "name": "device01", "id": "device_code", "type": "camera",
                                  "naming": "naming", "factory": "bb", "longitude": 222.98, "latitude": 124.89},
                       "captureTime": time, "captureImage": captureImg, "imageType": "jpg", "rowKey": "rowKey"}
            # print messgae
            try:
                sender.send('face-capture', json.dumps(messgae, ensure_ascii=False))
                print 'send to kafka successfully'
            except Exception as e:
                print e
                print 'Kafka send error ,save in local file system'
                cv2.imwrite(str(time) + '.jpg', res)


def face_dector_test(filename):
    image = cv2.imread("test.jpg")
    print image
    has_face(image)


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
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    producer = KafkaProducer(bootstrap_servers=['master:9092', 'node1:9092', 'node2:9092'])
    # client = KafkaClient(hosts=['192.168.31.72:9092','192.168.31.71:9092','192.168.31.73:9092'])
    # topic = client.topics['face-capture']
    # producer = topic.get_producer()
    rtsp = "rtsp://192.168.31.8:554/0000000000200000000000000530003:0000000000140000000000000500003:192.168.31.8:420000/MainStream"
    cap = cv2.VideoCapture(rtsp)
    ret = cap.read()
    while ret:
        frame = cap.read()
        # frame is a tuple,so use frame[1]
        Timer(0.04, capture(frame[1], time.time(), producer))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    # target = cv2.imread("test.jpg")
    # face_feature(target, detector, shape_predictor, face_rec_model)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
