import os
import cv2
import numpy as np
import argparse
import warnings
import time
from utils.custom_utils import detect_face, predict_fas, tracking

warnings.filterwarnings('ignore')

label, color = False, (0, 0, 0)

video = "../test_fas_video.mp4"


# Create a VideoCapture object called cap
cap = cv2.VideoCapture(0)

# This is an infinite loop that will continue to run until the user presses the `q` key
count_frame = 0
while cap.isOpened():
    count_frame += 1
    tic = time.time()

    # Read a frame from the webcam
    ret, frame_root = cap.read()
    frame_root = cv2.flip(frame_root, 1)
    frame = frame_root.copy()

    # If the frame was not successfully captured, break out of the loop
    if ret is False:
        break
    
    # Region for ID card 
    cv2.line(frame_root, (0, 400), (600, 400), (255,255,255), 2)
    cv2.line(frame_root, (600, 0), (600, 400), (255,255,255), 2)
    cv2.rectangle(frame, (0, 0), (600, 400), (255,255,255), -1)

    card_frame = frame_root[0:400, 0:600]
    card_frame = cv2.resize(card_frame, (0,0), fx=2, fy=2)

    card_face_bbox, conf =  detect_face(card_frame)
    if conf > 0.5:
        cv2.rectangle(card_frame, (card_face_bbox[0], card_face_bbox[1]), (card_face_bbox[0] + card_face_bbox[2], card_face_bbox[1] + card_face_bbox[3]), (0, 255, 0), 2)
    card_frame = cv2.resize(card_frame, (0,0), fx=0.5, fy=0.5)
    frame_root[0:400, 0:600] = card_frame

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tic = time.time()
    image_bbox, _ = detect_face(frame)
    new_register = tracking(image_bbox, frame_root)

        
        

    # print('Detection time: ', (time.time() - tic))
    if count_frame % 3 == 0:
        label, value = predict_fas(image_bbox, frame)

    
        
    test_speed = time.time() - tic
    fps = 1/test_speed

    if label:
        if label == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
    
    
    cv2.rectangle(frame_root, (image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)

    # Display the FPS on the frame
    cv2.putText(frame_root, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame on the screen
    cv2.imshow("frame", frame_root)
    # cv2.imshow("card", card_frame)


    # Check if the user has pressed the `q` key, if yes then close the program.
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the VideoCapture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()