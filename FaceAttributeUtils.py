import os
import numpy as np
import shutil

caffe_root = '/home/benny/caffe/'
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe


def predict(base64str):
    mean_filename = '/home/benny/caffe/models/age_gender_caffe/mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    age_net_pretrained = '/home/benny/caffe/models/age_gender_caffe/age_net.caffemodel'
    age_net_model_file = '/home/benny/caffe/models/age_gender_caffe/deploy_age.prototxt'
    age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                               mean=mean,
                               channel_swap=(2, 1, 0),
                               raw_scale=255,
                               image_dims=(256, 256))

    gender_net_pretrained = '/home/benny/caffe/models/age_gender_caffe/gender_net.caffemodel'
    gender_net_model_file = '/home/benny/caffe/models/age_gender_caffe/deploy_gender.prototxt'
    gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                  mean=mean,
                                  channel_swap=(2, 1, 0),
                                  raw_scale=255,
                                  image_dims=(256, 256))

    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Man', 'Female']
    gender_folder = 'male'
    input_image = caffe.io.load_image("/home/benny/PycharmProjects/video-project/face_test/test.jpg")
    input_image = base64str
    prediction = age_net.predict([input_image])
    print 'predicted age:', age_list[prediction[0].argmax()]
    prediction = gender_net.predict([input_image])
    print 'predicted gender:', gender_list[prediction[0].argmax()]
    return age_list[prediction[0].argmax()],gender_list[prediction[0].argmax()]


