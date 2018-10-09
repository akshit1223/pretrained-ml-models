#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: akshitbudhraja
"""

import os
import cv2
import random
import numpy as np
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
from xml.etree import ElementTree
from utils import label_map_util
from utils import visualization_utils as vis_util

base_data_path = os.getcwd() + '/data/RoadDamageDataset/'
save_data_path = os.getcwd() + '/test_runs/'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)
    

################# govs corresponds to municipality name ###################
govs = ["Adachi", "Chiba", "Ichihara", "Muroran", "Nagakute", "Numazu", "Sumida"]

damageTypes=["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44"]

####################### data exploration ########################
def draw_images(image_file):
    gov = image_file.split('_')[0]
    img = cv2.imread(base_data_path + gov + '/JPEGImages/' + image_file.split('.')[0] + '.jpg')
    
    infile_xml = open(base_data_path + gov + '/Annotations/' +image_file)
    tree = ElementTree.parse(infile_xml)
    root = tree.getroot()
    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # put text
        cv2.putText(img,cls_name,(xmin,ymin-10),font,1,(0,255,0),2,cv2.LINE_AA)

        # draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0),3)
    return img

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
def example_bounding_boxes():
    for ind, damageType in enumerate(damageTypes):
        tmp = []
        for gov in govs:
            file = open(base_data_path + gov + '/ImageSets/Main/%s_trainval.txt' %damageType, 'r')
    
            for line in file:
                line = line.rstrip('\n').split('/')[-1]
    
                if line.split(' ')[2] == '1':
                    tmp.append(line.split(' ')[0]+'.xml')
        random.shuffle(tmp)
        for number, image in enumerate(tmp[0:1]):
            img = draw_images(image)
            image_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_data_path + '/example_' + str(ind) + '.jpg', image_to_write)

#################### running pretrained model #########################

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  os.getcwd() + '/models/ssd_mobilenet_RoadDamageDetector.pb' 

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.getcwd() + '/models/crack_label_map.pbtxt'

NUM_CLASSES = 8

def run_model(test_images_dir):
    if not os.path.exists(test_images_dir):
        print("Invalid directory")
        return
    if not os.path.exists(os.path.join(test_images_dir, 'results')):
        os.makedirs(os.path.join(test_images_dir, 'results'))
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    TEST_IMAGE_PATHS= [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f)) and f.find('.jpg') > -1]

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for ind, image_path in enumerate(TEST_IMAGE_PATHS):
          image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              min_score_thresh=0.3,
              use_normalized_coordinates=True,
              line_thickness=8)
          image_to_write = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(test_images_dir, 'results') + '/example_' + str(ind) + '.jpg', image_to_write)

print("Running model on training examples . . .")
example_bounding_boxes() 
test_images_dir = os.path.join(os.getcwd(), 'test_runs', 'test_images')
print("Running model on test images in " + str(test_images_dir) + " . . .")
run_model(test_images_dir)
