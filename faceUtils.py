# -*- coding: UTF-8 -*-
import cv2
import dlib
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


def face_feature(image_array):
    face_cuts = face_cut(image_array)
    for image_array in face_cuts:
        img = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        dets = detector(img, 1)
        face_features = []
        for index, face in enumerate(dets):
            print('face {};left:{}top{};right{};bootpm {}'.format(index, face.left(), face.top(), face.right(),
                                                                  face.bottom()))
            shape = shape_predictor(img2, face)
            face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)
            vectors = dlib_vector_to_array(face_descriptor)
            import FaceAttributeUtils
            age ,gender = FaceAttributeUtils.predict(img)
            face_features.append({"feature":vectors,"gender":gender,"age":age})
    return face_features


def dlib_vector_to_array(face_descriptor):
    vectors = np.array([])
    for i, num in enumerate(face_descriptor):
        vectors = np.append(vectors, num)
    import ImageUtils
    return ImageUtils.byte_to_base64(vectors.tobytes())

def face_cut(image_array):
    face_cuts = []
    img = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
    dets = detector(img, 1)
    # image width
    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    # dlib detector
    for k, d in enumerate(dets):

        # 计算矩形框大小
        height = d.bottom() - d.top()
        width = d.right() - d.left()

        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            if d.top() + i >= img_height:  # 防止越界
                continue
            for j in range(width):
                if d.left() + j >= img_width:
                    continue
                img_blank[i][j] = img[d.top() + i][d.left() + j]
        img_blank = cv2.resize(img_blank, (200, 200), interpolation=cv2.INTER_CUBIC)
        face_cuts.append(cv2.imencode('.jpg', img_blank)[1])
    return face_cuts


