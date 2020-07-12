#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util
from datetime import datetime
import uuid
import dlib
import face_recognition
import facenet
import mysql.connector

from functions import *

from scipy import misc

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

pathtofacenet = './model/testmodel/20180402-114759.pb'

pose_predictor_68_point = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('./model/dlib_face_recognition_resnet_model_v1.dat')



NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

minsize = 20
input_image_size = 160

# mydb = mysql.connector.connect(
#   host="127.0.0.1",
#   port='3311',
#   user="root",
#   passwd="Amitsaini374",
#   auth_plugin='sha256_password',
#   database="Aknownymousdb"
# )
# mycursor = mydb.cursor()


# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk

# Get input and output tensors


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.

        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time

        print('inference time cost: {}'.format(elapsed_time))


        return (boxes, scores, classes, num_detections)


class TensoflowFaceReckon(object):
    """docstring for ClassName"""
    def __init__(self, pathtofacenet):
        facenet.load_model("./model/testmodel/20180402-114759.pb")
        self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        self.sess = tf.compat.v1.Session()
                



    def getEmbedding(self, resized):
        self.reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
        self.feed_dict = {self.images_placeholder: self.reshaped, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.embeddings, feed_dict=self.feed_dict)
        return embedding








if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print ("usage:%s (cameraID | filename) Detect faces\
 in the video example:%s 0"%(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
      camID = int(sys.argv[1])
    except:
      camID = sys.argv[1]
    
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    tReckon = TensoflowFaceReckon(pathtofacenet)


    cap = cv2.VideoCapture(camID)
    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        #print (h, w)

        (boxes, scores, classes, num_detections) = tDetector.run(image)



        [height, width] = image.shape[:2]

        max_boxes_to_draw = boxes.shape[0]

        z= np.squeeze(boxes)
        s= np.squeeze(scores)
        c= np.squeeze(classes).astype(np.int32)
        n=np.squeeze(num_detections).astype(np.int32)
        #print(c)

        max_box=z.shape[0]


        

        for i in range(max_box):
            if s[i] > 0.75 and c[i]==1:
                bx = z[i]
                ymin = int(bx[0]*height)
                xmin = int(bx[1]*width)
                ymax = int(bx[2]*height)
                xmax = int(bx[3]*width)

                if((xmax-xmin)>10 and (ymax-ymin)>10 and bx.any() ):

                    # this function is for the first image

                    t1 = cv2.imread('amit5.jpg')
                    resize1 = cv2.resize(t1, (160, 160))
                    resize12=facenet.prewhiten(resize1)
                    emb2= tReckon.getEmbedding(resize1)

                    # this function is for the second image


                    cv2.imshow('input1', resize1)
                    #print(emb2)

                    # t2 = cv2.imread('amit7.jpg')
                    # resize1 = cv2.resize(t2, (160, 160))
                    # resize13=facenet.prewhiten(resize1)
                    # emb3= tReckon.getEmbedding(resize13)



                    
                    # this function is for the live feed delivered to the python function


                    Result = np.array(image[ymin:ymax,xmin:xmax])
                    a= uuid.uuid4()
                    inimg = image[ymin-30:ymax+30,xmin-30:xmax+30]
                    resize15 = cv2.resize(inimg, (160, 160))
                    resize=facenet.prewhiten(resize15)



                    cv2.imshow('input2', resize15)
                    #cv2.imwrite("E:/aknownymous/tensorflow-face-detection-master/images/" + str(a) + ".jpg", resize15)
                    


                    emb1= tReckon.getEmbedding(resize15)

                    rez = np.sum(emb2*emb1,axis=1)
                    print('cosine')

                    print(rez)
                    print('eculidian')

                    dist = np.sqrt(np.sum(np.square(np.subtract(emb1[0,:],emb2[0,:]))))
                    print(dist)

                    # sql = "SELECT * FROM aknownymousdb.id_encoding_table"
                    # mycursor.execute(sql)

                    # myresult = mycursor.fetchall()

                    # for x in myresult:
                    #   print(x)









        if windowNotSet is True:
            #cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        #cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
