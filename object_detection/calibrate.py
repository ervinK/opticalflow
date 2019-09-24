import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import math
import givepoints
import time
import pickle




from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from sparseFlow import Sparse
from numpy.linalg import inv

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


from utils import label_map_util

from utils import visualization_utils as vis_util




# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map_after_09.07.pbtxt')


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, sess, graph):
  with graph.as_default():
    
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[1], image.shape[2])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.3), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

objx1, objx2, objy1, objy2 = -1, -1, -1, -1
roix1, roix2, roix3, roix4, roiy1, roiy2, roiy3, roiy4 = -1, -1, -1, -1, -1, -1, -1, -1

counter = 0
counterROI = 0
def givePointsOfRealObject(event,x,y,flags,param):
  global objx1, objx2, objy1, objy2
  global counter
  if event == cv2.EVENT_LBUTTONDOWN:
    print("VALOS OBJEKTUM KIJELOLESE")
    if counter == 0:
        objx1, objy1 = x, y
        
        counter += 1
    elif counter == 1:
        objx2, objy2 = x, y
        counter += 1
    else:
        givepoints.alertObject(objx1, objy1, objx2, objy2)
        return

def givePointsOfROI(event,x,y,flags,param):
  global roix1, roix2, roix3, roix4, roiy1, roiy2, roiy3, roiy4
  global counterROI
  if event == cv2.EVENT_LBUTTONDOWN:
    print("ROI KIJELOLESE")
    if counterROI == 0:
      roix1, roiy1 = x, y
      counterROI += 1
    elif counterROI == 1:
      roix2, roiy2 = x, y
      counterROI += 1
    elif counterROI == 2:
      roix3, roiy3 = x, y
      counterROI += 1
    elif counterROI == 3:
      roix4, roiy4 = x, y
      counterROI += 1
    else:
      givepoints.alertROI(roix1, roiy1, roix2, roiy2, roix3, roiy3, roix4, roiy4)
      return





sess = tf.Session(graph=detection_graph, config=config) 







settingImg2 = cv2.imread("videos/AP2.jpg")

transX = 1000
transY = settingImg2.shape[0] * (transX/settingImg2.shape[1])

settingImg2 = cv2.resize(settingImg2, (int(transX), int(transY)))


cv2.namedWindow('image2')
cv2.setMouseCallback('image2',givePointsOfROI)
cv2.imshow('image2', settingImg2)
cv2.waitKey(0)
cv2.destroyAllWindows()


sorted_points = givepoints.sortGivenPoints([roix1, roiy1], [roix2, roiy2], [roix3, roiy3], [roix4, roiy4])

cv2.imshow('image', givepoints.printROIPoints(settingImg2, sorted_points[0][0], sorted_points[0][1], sorted_points[1][0], sorted_points[1][1], sorted_points[2][0], sorted_points[2][1], sorted_points[3][0], sorted_points[3][1]))
cv2.waitKey(0)
cv2.destroyAllWindows()


processImg = cv2.imread("videos/AP2.jpg")

processImg = cv2.resize(processImg, (int(transX), int(transY)))



roix1, roiy1, roix2, roiy2, roix3, roiy3, roix4, roiy4 = sorted_points[0][0], sorted_points[0][1], sorted_points[1][0], sorted_points[1][1], sorted_points[2][0], sorted_points[2][1], sorted_points[3][0], sorted_points[3][1]

points1 = np.float32([[roix1, roiy1], [roix2, roiy2], [roix3, roiy3], [roix4, roiy4]])
        
points2 = np.float32([[0,0], [processImg.shape[1],0], [0, processImg.shape[0]], [processImg.shape[1], processImg.shape[0]]])

matrix = cv2.getPerspectiveTransform(points1, points2) #transzformacios matrix

print(matrix)


ct_frame = cv2.warpPerspective(processImg, matrix, (processImg.shape[1], processImg.shape[0]))

settingImg = ct_frame
cv2.namedWindow('image')
cv2.setMouseCallback('image',givePointsOfRealObject)
cv2.imshow('image', settingImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image', givepoints.printObjectPoints(settingImg, objx1, objy1, objx2, objy2))
cv2.waitKey(0)
cv2.destroyAllWindows()




lineSize = math.sqrt(abs(math.pow(objx2-objx1,2) + math.pow(objy2-objy1,2))) 

print(lineSize)

real_object_size = input('Real size of the object(m):\n')

print(real_object_size)

d = (matrix, lineSize, real_object_size, (transX, transY))

with open('datas.pickle', 'wb') as f:
    pickle.dump(d, f)