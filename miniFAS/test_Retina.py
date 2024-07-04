import os
import cv2
import warnings
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

from function_model.fas import FaceAntiSpoofing


warnings.filterwarnings("ignore")

dataset = "datasets/Test_SCI"

fas1_normal_path = "model_onnx/2.7_80x80_MiniFASNetV2.onnx"
fas2_normal_path = "model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"

fas1_normal = FaceAntiSpoofing(fas1_normal_path)
fas2_normal = FaceAntiSpoofing(fas2_normal_path)


if __name__ == "__main__":
 
    count_none_face = 0
    count_llie = 0
    count_undefined = 0
    
   

    images = os.listdir(dataset)
   
    TIME_START = time.time()
    for image in tqdm(images):
        prediction = np.zeros((1, 3))
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)  # BGR


        pred1 = fas1_normal.predict(img)
        pred2 = fas2_normal.predict(img)
            
        if  pred1 is None or pred2 is None:
            count_none_face += 1    
           
    acc_Rent = (1048-count_none_face)/1048
   
    print("Accuracy Rentina:", acc_Rent*100)
    print("**************")
    print("count none face: ", count_none_face)
    print("time: ", time.time() - TIME_START)
    