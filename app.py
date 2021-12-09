import cv2
import argparse

import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
from utils import label_map_util as label_map_utils
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
               default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 1
color = (255, 0, 0)

detection_graph, sess = detector_utils.load_inference_graph()




if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    vs = VideoStream(0).start()
    # Oriendtation of machine0
    Orientation = 'bt'
    # max number of objects we want to detect/track
    detect = 4

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    count=1


    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'Output_{start_time}.avi', fourcc, 10, (640, 480))
        while True:
            frame = vs.read()
            frame = np.array(frame)
            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # Draw bounding boxes and text
            detector_utils.draw_box_on_image(
                detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)
            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:

                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

    except KeyboardInterrupt:
        print("error")
