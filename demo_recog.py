import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from utils.custom_utils import detect_face, predict_fas, tracking, get_feature, cosine_similarity, crop_face

def verify(frame_queue, result_queue):
    while True:
        # Get frame from the queue
        images = frame_queue.get()
        if len(images) == 0:
            continue
        elif len(images) == 2:
            feature1 = get_feature(images[0])
            feature2 = get_feature(images[1])
            similarity = cosine_similarity(feature1, feature2)
            if similarity >= 0.75:
                result_queue.put("MATCH ({})".format(similarity))
            else:
                result_queue.put("NOT_MATCH ({})".format(similarity))

                
def camera(frame_fas, result_fas, frame_verify, result_verify):
    batch_face = []
    start_fas = False
    verified = False
    card = False
    label, color = False, (0, 0, 0)

    # Create a VideoCapture object called cap
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # This is an infinite loop that will continue to run until the user presses the `q` key
    count_frame = 0
    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame_root = cap.read()
        if ret is False:
            break

        frame_root = cv2.flip(frame_root, 1)
        frame = frame_root.copy()

        # Region for ID card 
        cv2.rectangle(frame_root, (0,0), (500,300), (255,255,255), 2)
        cv2.rectangle(frame, (0,0), (500,300), (255,255,255), -1)


        card_frame = frame_root[0:300, 0:500]
        card_frame = cv2.resize(card_frame, (0,0), fx=2, fy=2)

        card_face_bbox, conf1 =  detect_face(card_frame)
        
        if conf1 >= 0.2 and card_face_bbox != [0,0,1,1]:
            cv2.rectangle(card_frame, (card_face_bbox[0], card_face_bbox[1]), (card_face_bbox[0] + card_face_bbox[2], card_face_bbox[1] + card_face_bbox[3]), (255, 255, 102), 2)
        else:
            card_face_bbox = None
        #######################################

        face_bbox, _ = detect_face(frame)

        if card_face_bbox is not None:
            if not count_time :
                count_time = time.time()
            
            if time.time() - count_time < 4:
                cv2.putText(card_frame, str(int(time.time() - count_time)), (950,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5, cv2.LINE_AA)
            elif same_person:
                cv2.putText(card_frame, str(same_person), (700,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5, cv2.LINE_AA)
                
            
            if not verified and (time.time() - count_time)>3 :
                face = crop_face(frame, face_bbox)
                card_face = crop_face(card_frame, card_face_bbox)
                frame_verify.put([face, card_face])
                verified = True
        else:
            verified = False
            count_time = False
            same_person = False


        if not result_verify.empty():
            same_person = result_verify.get()
            print(same_person)

        if start_fas:
            batch_face.append((face_bbox, frame))
            count_frame += 1
        
        if count_frame == 5:
            frame_fas.put(batch_face)
            count_frame = 0
            batch_face = []
            start_fas = False

        if not result_fas.empty():
            label = result_fas.get()

        new_gister = tracking(face_bbox, frame_root)
        # new_card = tracking(card_face_bbox, card_frame)

        if new_gister:
            start_fas = True
            verified = False
        

        if label:
            if label == "REAL":
                color = (0, 255, 0)
                not_spoof = True
            elif label == "FAKE":
                color = (0, 0, 255)
                not_spoof = False
        
        
        card_frame = cv2.resize(card_frame, (0,0), fx=0.5, fy=0.5)
        frame_root[0:300, 0:500] = card_frame
        cv2.rectangle(frame_root, (face_bbox[0], face_bbox[1]), (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]), color, 2)


        cv2.putText(frame_root, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame_root)

        # Check if the user has pressed the `q` key, if yes then close the program.
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()

    # Close all open windows
    cv2.destroyAllWindows()

def anti_spoofing(frame_queue, result_queue):
    while True:
        real, fake = 0, 0

        # Get frame from the queue
        detections = frame_queue.get()

        for (bbox, frame) in detections:
            frame = np.asarray(frame, dtype=np.uint8) 
            
            label, value = predict_fas(bbox, frame)

            if label == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
            result_queue.put("REAL")
        else:
            result_queue.put("FAKE")

if __name__ == "__main__":
    frame_verify = multiprocessing.Queue()
    result_verify = multiprocessing.Queue()
    frame_fas = multiprocessing.Queue()
    result_fas = multiprocessing.Queue()
    print("LOADING .......")
    p1 = multiprocessing.Process(name='p1', target=camera, args=(frame_fas, result_fas, frame_verify, result_verify))
    p2 = multiprocessing.Process(name='p2', target=anti_spoofing, args=(frame_fas, result_fas))
    p3 = multiprocessing.Process(name='p3', target=verify, args=(frame_verify, result_verify))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    
