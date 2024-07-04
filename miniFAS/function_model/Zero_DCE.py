import cv2
import onnxruntime
import torch
import torch.optim
import numpy as np
import torchvision


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

class LowLightEnhancer:
    def __init__(self, scale_factor=1, model_onnx="miniFAS/model_onnx/ZeroDCE++.onnx"):
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
        image = image[:, :, ::-1]

        img_processed = image / 255.0
        img_processed = torch.from_numpy(img_processed).float()
        h = (img_processed.shape[0] // self.scale_factor) * self.scale_factor
        w = (img_processed.shape[1] // self.scale_factor) * self.scale_factor
        img_processed = img_processed[0:h, 0:w, :]
        img_processed = img_processed.permute(2,0,1)
        img_processed = img_processed.unsqueeze(0)
        return img_processed

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
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        rgb_image = torch.from_numpy(rgb_image)
        grid = torchvision.utils.make_grid(rgb_image)
        rgb_image = (
            grid.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
        )
        return rgb_image[:, :, ::-1]
