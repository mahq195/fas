import os
import cv2
import cv2 as cv
import numpy as np

import warnings
import torch
from torchvision import transforms
from PIL import Image

from tracker.centroid_tracker import CentroidTracker

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'

model_test = AntiSpoofPredict(0)

ct = CentroidTracker()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
transformer = data_transforms['val']


def detect_face(image):
    image_bbox, conf = model_test.get_bbox(image)
    # if conf < 0.5:
    #     image_bbox = None
    return image_bbox, conf


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

def adjust_bounding_box(box):
    x, y, w, h = box
    if h > w:
        diff = h - w
        w = h
        x -= int(diff // 2)  
        x = max(x, 0)
        w = h
    return x, y, w, h

def load_model():
    scripted_model_file = 'mobilefacenet/mobilefacenet_scripted.pt'
    model = torch.jit.load(scripted_model_file)
    model = model.to(device)
    model.eval()
    return model

def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img

extract_model = load_model()
def get_feature(img, model=extract_model):
    img = cv2.resize(img, (112, 112))
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    imgs[0] = transform(img.copy(), False)
    imgs[1] = transform(img.copy(), True)
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity

def crop_face(image, bbox):
    bbox = adjust_bounding_box(bbox)
    x, y, w, h = bbox
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min((x+w), image.shape[1])
    y2 = min((y+h), image.shape[0])
    face = image[y1:y2, x1:x2]
    return face