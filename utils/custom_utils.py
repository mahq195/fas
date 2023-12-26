import os
import cv2
import cv2 as cv
import numpy as np

import warnings

from tracker.centroid_tracker import CentroidTracker

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'

model_test = AntiSpoofPredict(0)

ct = CentroidTracker()


def detect_face(image):
    image_bbox, conf = model_test.get_bbox(image)
    # if conf < 0.5:
    #     image_bbox = None
    return image_bbox, conf

def process_model1(image_bbox, frame):
    model_name = model_1.split('/')[-1]
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    param = {
        "org_img": frame,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img, img_ = CropImage().crop(**param)
    prediction = model_test.predict(img, model_1)
    return prediction

def process_model2(image_bbox, frame):
    model_name = model_2.split('/')[-1]
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    param = {
        "org_img": frame,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img, img_ = CropImage().crop(**param)
    prediction = model_test.predict(img, model_2)
    return prediction


def predict_fas(image_bbox, frame):
    prediction = np.zeros((1, 3))
    
    for model in [model_1, model_2]:
        model_name = model.split('/')[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img, img_ = CropImage().crop(**param)
        prediction += model_test.predict(img, model)

    label = np.argmax(prediction)
    value = prediction[0][label]
    return label, value

def tracking(bbox, frame):
    # bbox = (x, y, w, h)
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = (x, y, x+w, y+h)
    if bbox == (0, 0, 1, 1):
        objects, new_register = ct.update([])
    else:
        objects, new_register = ct.update([tuple(bbox)])

    # for (objectID, centroid) in objects.items():
    #     # draw both the ID of the object and the centroid of the
    #     # object on the output frame
    #     text = "ID {}".format(objectID)
    #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 0), -1)

    return new_register