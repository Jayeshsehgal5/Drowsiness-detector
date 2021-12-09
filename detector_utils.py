import numpy as np
import tensorflow as tf
from playsound import playsound
import cv2
from utils import label_map_util
detection_graph = tf.Graph()
TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'
k=[]
NUM_CLASSES = 4
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2
# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

l=[]
def draw_box_on_image(detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    focalLength = 875
    avg_width = 4.0
    count=0
    color = None
    color0 = (255, 100, 0)
    color1 = (255, 100, 0)
    for i in range(detect):

        if (scores[i] > score_thresh):
            if classes[i] == 1:
                id = 'Open Eyes'
                # b=1

            if classes[i] == 2:
                id = 'Close Eyes'
            if classes[i] == 3:
                id = 'Yawn'

            if classes[i] == 4:
                id = 'Not Yawning'

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            #dist = distance_to_camera(avg_width, focalLength, int(right - left))

            #if dist:
                #hand_cnt = hand_cnt + 1
            cv2.rectangle(image_np, p1, p2, color, 3, 1)
            if id == "Close Eyes":
                cv2.putText(image_np, id + str(i) + ': ' + id, (int(left), int(top) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                l.append(1)
                print(len(l))
                if len(l)>=5:
                    playsound("utils/alarm.mp3")
                    cv2.putText(image_np, '******ALERT******', (50, 50), 2, 1, 255, 1, cv2.LINE_AA)
                    l.clear()
            elif id=="Yawn":
                cv2.putText(image_np, id + str(i) + ': ' + id, (int(left), int(top) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                k.append(2)
                if len(k)>3:
                    cv2.putText(image_np, '******YOU ARE DROWSY******', (50, 50), 2, 1, 255, 1, cv2.LINE_AA)
                    k.clear()



            else:
                cv2.putText(image_np, id + str(i) + ': ' + id, (int(left), int(top) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
                        #(int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

    return count


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})


    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
