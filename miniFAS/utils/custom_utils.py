import os
import cv2
import cv2 as cv
import numpy as np

import warnings
import torch
from torchvision import transforms
from PIL import Image

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_1 = 'miniFAS/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'miniFAS/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'

model_test = AntiSpoofPredict(0)


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
    if conf < 0.7:
        image_bbox = None
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


def adjust_bounding_box(box):
    x, y, w, h = box
    if h > w:
        diff = h - w
        w = h
        x -= int(diff // 2)  
        x = max(x, 0)
        w = h
    return x, y, w, h


def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def crop_face(image, bbox):
    bbox = adjust_bounding_box(bbox)
    x, y, w, h = bbox
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min((x+w), image.shape[1])
    y2 = min((y+h), image.shape[0])
    face = image[y1:y2, x1:x2]
    return face