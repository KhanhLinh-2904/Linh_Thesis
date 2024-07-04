import random
import cv2
import onnxruntime
import torch
import torch.optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

class LowLightEnhancer:
    def __init__(self, scale_factor=1, model_onnx="miniFAS/model_onnx/SCI.onnx"):
        self.scale_factor = scale_factor
        self.transform = torchvision.transforms.ToTensor()
        self.ort_session = onnxruntime.InferenceSession(
            model_onnx, providers=["CPUExecutionProvider"]
        )

    def get_threshold(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_intensity = cv2.mean(gray_image)[0]
        return average_intensity

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        img_norm = transform(image).numpy()

        low = np.transpose(img_norm, (1, 2, 0))
        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))
        low = torch.from_numpy(low)
        C,H,W = low.shape
        low = torch.reshape(low, (1, C, H,W))
        return low

    def enhance(self, image):
        img_processed = self.preprocess(image)
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(img_processed)}
        onnxruntime_outputs = self.ort_session.run(None, ort_inputs)
        output1 = onnxruntime_outputs[0].reshape(
            onnxruntime_outputs[0].shape[1],
            onnxruntime_outputs[0].shape[2],
            onnxruntime_outputs[0].shape[3],
        )

        red_channel = output1[0]
        green_channel = output1[1]
        blue_channel = output1[2]
        rgb_image = np.stack([blue_channel, green_channel, red_channel ], axis=-1) #BGR
        rgb_image = Image.fromarray(np.clip(rgb_image * 255.0, 0, 255.0).astype('uint8'))
        # img = img[:, :, ::-1]
        # rgb_image.save('ex0.jpg')
        # print("rgb_image: ",rgb_image)
        img = np.asarray(rgb_image)
        
        return img
