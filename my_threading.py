import cv2
import multiprocessing
import numpy as np
from utils.custom_utils import detect_face, predict_fas, tracking, process_model1, process_model2






def anti_spoofing(frame_queue, result_queue):
    while not frame_queue.empty():
        real, fake = 0, 0

        # Get frame from the queue
        detections = frame_queue.get()
        for (frame, bbox) in detections:
            # Create a multiprocessing pool with 2 processes
            with multiprocessing.Pool(processes=2) as pool:

                # Apply model 1 to the data using the first process
                result_model1 = pool.map(process_model1, bbox, frame)

                # Apply model 2 to the data using the second process
                result_model2 = pool.map(process_model2, bbox, frame)

            final_result = result_model1 + result_model2
            label = np.argmax(final_result)
        
            if label == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
            result_queue.put("REAL")
            print("REAL")
        else:
            result_queue.put("FAKE")
            print("FAKE")

def main(q, r):
    # Initialize multiprocessing components
    # frame_queue = multiprocessing.Queue()
    # result_queue = multiprocessing.Queue()

    # # Start anti-spoofing process
    # anti_spoofing_process = multiprocessing.Process(target=anti_spoofing, args=(frame_queue, result_queue))
    # anti_spoofing_process.start()

    # Main thread for capturing frames
    cap = cv2.VideoCapture(0)
    frame_count = 0
    detections = []

    while True:
        # if not q.empty():
        #     print(r.get())
        ret, frame = cap.read()
        frame_count += 1
        q.put(frame)
        if not r.empty():
            print(r.get())
        # print("put")
        # Perform face detection every frame
        image_bbox, _ = detect_face(frame)
        # detections.append((frame, image_bbox))

        # Check anti-spoofing result after 10 frames
        # if frame_count == 10:
        #     frame_queue.put(detections)

            # Reset frame count
            # frame_count = 0
            # detections = []

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Terminate the anti-spoofing process
    # anti_spoofing_process.terminate()

def now(q, r):
    t = False
    while True:
        frame = q.get()
        if t == False:
            r.put("REAL")
            t = True
        # r.put("REAL")
        print("get")
        image = np.asarray(frame, dtype=np.uint8) 
        
        
        
        

if __name__ == "__main__":
    q = multiprocessing.Queue()
    r = multiprocessing.Queue()
    p1 = multiprocessing.Process(name='p1', target=main, args=(q, r))
    p2 = multiprocessing.Process(name='p2', target=now, args=(q, r))
    p1.start()
    p2.start()
    p1.join()
    p2.join()