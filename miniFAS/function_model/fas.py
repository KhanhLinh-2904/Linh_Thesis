import numpy as np
import warnings
import torch


from src.anti_spoof_predict import AntiSpoofPredict, Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans
import onnxruntime
import torch.nn.functional as F
warnings.filterwarnings('ignore')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class FaceAntiSpoofing:

    def __init__(self, model_onnx="miniFAS/model_onnx/2.7_80x80_MiniFASNetV2"):
        self.model_path = model_onnx
        self.ort_session = onnxruntime.InferenceSession(
            model_onnx, providers=["CPUExecutionProvider"]
        )
    def get_bbox_face(self, image):
        face_detection = Detection()
        image_bbox, conf = face_detection.get_bbox(image)
        if conf < 0.7:
            return None
        else:
            return image_bbox

    def preprocess(self, image, image_bbox):
        image_cropper = CropImage()
        model_name = self.model_path.split("/")[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).cpu()
        return img

    def predict(self, image):
        image_bbox = self.get_bbox_face(image)
        if image_bbox is None:
            return None
        img_processed = self.preprocess(image, image_bbox)
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(img_processed)}
        # print('ort_inputs.shape: ', len(ort_inputs))
        # print('ort_inputs.shape: ', ort_inputs.values)
        onnxruntime_outputs = self.ort_session.run(None, ort_inputs)
        onnxruntime_outputs = torch.Tensor(onnxruntime_outputs)
        onnxruntime_outputs = onnxruntime_outputs.view(1,3)
        onnxruntime_outputs = F.softmax(onnxruntime_outputs).cpu().numpy()
        return onnxruntime_outputs
        

