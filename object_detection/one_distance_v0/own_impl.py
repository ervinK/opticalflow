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





# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/label_map_after_09.07.pbtxt'
PATH_TO_FROZEN_GRAPH = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_new.pb'




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





cap = cv2.VideoCapture("videos/measure/hid_elol_50.mp4")


#-------------------------------------------------------------------------------

transX = 1500
transY = cap.read()[1].shape[0] * (transX/cap.read()[1].shape[1])

#-------------------------------------------------------------------------------


settingImg2 = cap.read()[1]
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


processImg = cap.read()[1]
processImg = cv2.resize(processImg, (int(transX), int(transY)))
#points1 = np.float32([[1028, 253],[1215, 269],[192, 526],[557, 668]])

roix1, roiy1, roix2, roiy2, roix3, roiy3, roix4, roiy4 = sorted_points[0][0], sorted_points[0][1], sorted_points[1][0], sorted_points[1][1], sorted_points[2][0], sorted_points[2][1], sorted_points[3][0], sorted_points[3][1]

points1 = np.float32([[roix1, roiy1], [roix2, roiy2], [roix3, roiy3], [roix4, roiy4]])
        
points2 = np.float32([[0,0], [processImg.shape[1],0], [0, processImg.shape[0]], [processImg.shape[1], processImg.shape[0]]])

matrix = cv2.getPerspectiveTransform(points1, points2) #transzformacios matrix



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

real_object_size = input('Real size of the object(m):\n')



font = cv2.FONT_HERSHEY_DUPLEX

feature_params = dict(maxCorners = 850, qualityLevel = 0.01, minDistance = 1.2, blockSize = 7)
lk_params = dict(winSize = (40, 40), maxLevel = 10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

lineSize = math.sqrt(abs(math.pow(objx2-objx1,2) + math.pow(objy2-objy1,2))) 



framelist = []
i = 0


framecounter = 0
start = time.time()

while(cap.isOpened()):

  try:
    ret, frame = cap.read()
    

    frame = cv2.resize(frame, (int(transX), int(transY)))
    default_frame = frame.copy()

    

    image_np_expanded = np.expand_dims(frame, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, sess, detection_graph)
    # Visualization of the results of a detection.

    height, width, depth = frame.shape

    boxes = output_dict['detection_boxes']
    probabs = output_dict['detection_scores']
    classes = output_dict['detection_classes']

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=5, min_score_thresh=0.2)

    framelist.append(default_frame) #frame hozzaadasa
    
    if i > 0: #2 framenkent hasonlitjuk ossze a valtozasokat
      obj = Sparse(framelist, matrix)
      points, threed_points = obj.evaluate(feature_params, lk_params, transX, transY)
      

      for x in range(0, len(boxes)):
        if(probabs[x] > 0.2) and classes[x] == 4:
          sum_of_distances = 0.0
          count_moves = 0
          avg_speed = 0
          
          ymin = int(boxes[x][0]*height)
          xmin = int(boxes[x][1]*width)
          ymax = int(boxes[x][2]*height)
          xmax = int(boxes[x][3]*width)
          if len(points) != 0:
            for p in range(0, len(points)):
              if ((threed_points[p][0][0] >= xmin*1.0 and threed_points[p][0][0] <= xmax*1.0) and (threed_points[p][0][1] >= ymin*1.0 and threed_points[p][0][1] <= ymax*1.0)):
                dis = math.sqrt(pow(points[p][0][0]-points[p][1][0], 2) + pow(points[p][0][1]-points[p][1][1], 2))
                spedd = 3.6*(dis/lineSize)*float(real_object_size)*30
                print(spedd)
                sum_of_distances += math.sqrt(pow(points[p][0][0]-points[p][1][0], 2) + pow(points[p][0][1]-points[p][1][1], 2))
                count_moves += 1
                frame = cv2.line(frame, (threed_points[p][0][0], threed_points[p][0][1]), (threed_points[p][1][0], threed_points[p][1][1]), (0, 255, 0), 2)
                frame = cv2.circle(frame, (threed_points[p][0][0], threed_points[p][0][1]), 3, (0, 255, 0), -1)
            if count_moves != 0:
              avg_speed = sum_of_distances/float(count_moves)
            print(str(3.6*((avg_speed/lineSize) * float(real_object_size))*30))
            frame = cv2.putText(frame, str(3.6*((avg_speed/lineSize) * float(real_object_size))*30) + "km/h", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                
      framelist = []
      i = 0
    else:
      i += 1
    framecounter += 1
    cv2.imshow('frame', frame)
    cv2.waitKey(0) 
  except:
    break

end = time.time() #feldolgozas befejezese

print(start)
print(end)
print(framecounter)

print(framecounter/(end-start))
  
 


  
