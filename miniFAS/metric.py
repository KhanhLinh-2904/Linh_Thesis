import os

import cv2
import numpy as np
import torch
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from scipy import signal
from PIL import Image 
from tqdm import tqdm




def check_empty_img(img): 
  
    image = cv2.imread(img) 
    if image is None: 
        result = "Image is empty!!"
    else: 
        result = "Image is not empty!!"
  
    return result 
      
def cal_mean_abs_diff(img1, img2):
    
    diff = cv2.absdiff(img1, img2)
    mean_absolute_difference = np.mean(diff)
    return mean_absolute_difference

def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map

def mse(imageA, imageB):
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    return mse_error

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def calculate_metrics(original_img, enhanced_img):
    original_img = original_img.astype(float) / 255.0
    enhanced_img = enhanced_img.astype(float) / 255.0
    original_img = cv2.resize(original_img, (enhanced_img.shape[1], enhanced_img.shape[0]))
    # PSNR
    psnr_value = psnr(original_img, enhanced_img, data_range=1)

    # SSIM
    ssim_value = ssim(original_img, enhanced_img, data_range=1)
    # Entropy
    entropy_original = entropy(original_img.flatten())
    entropy_enhanced = entropy(enhanced_img.flatten())

    # Standard Deviation
    std_dev_original = np.std(original_img)
    std_dev_enhanced = np.std(enhanced_img)
    original_tensor = torch.from_numpy(original_img)
    enhanced_tensor = torch.from_numpy(enhanced_img)

    mse_value = mse(original_img, enhanced_img)
    maf_value = cal_mean_abs_diff(original_img, enhanced_img)
    # print("maf_value: ",maf_value)
    return (
        psnr_value,
        ssim_value,
        entropy_original,
        entropy_enhanced,
        std_dev_original,
        std_dev_enhanced,
        mse_value,
        maf_value,
    )


if __name__ == '__main__':
    
    original_image_folder = 'Dataset/updated-SCI/result_test'
    enhanced_image_folder = 'Dataset/updated-SCI/result_test'

    images = os.listdir(enhanced_image_folder)
    psnr_arr = []
    ssim_arr = []
    entropy_original_arr = []
    entropy_enhanced_arr = []
    std_dev_original_arr = []
    std_dev_enhanced_arr = []
    mse_arr = []
    mean_absolute_differences = []
    results = ''
    niqe_arr = []
    for image in tqdm(images):
        enhanced_image_path = os.path.join(enhanced_image_folder, image)
        original_image_path = os.path.join(original_image_folder, image)
    #     label_image_name = image.split('_')[0] + '.jpg'
    #     original_image_path = os.path.join(original_image_folder, label_image_name)
    #     enhanced_image_path = os.path.join(enhanced_image_folder, image)
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None or enhanced_img is None:
            print("enhanced_img path None:", enhanced_image_path)
            print("original_img path None:", original_image_path)
            continue
        if original_img is not None:
            (
                psnr_value,
                ssim_value,
                entropy_original,
                entropy_enhanced,
                std_dev_original,
                std_dev_enhanced,
                mse_value,
                maf_value,
            ) = calculate_metrics(original_img, enhanced_img)
            psnr_arr.append(psnr_value)
            ssim_arr.append(ssim_value)
            entropy_original_arr.append(entropy_original)
            entropy_enhanced_arr.append(entropy_enhanced)
            std_dev_original_arr.append(std_dev_original)
            std_dev_enhanced_arr.append(std_dev_enhanced)
            mse_arr.append(mse_value)
            # print(type(maf_value))
            mean_absolute_differences.append(maf_value)


    psnr_value = sum(psnr_arr) / len(psnr_arr)
    ssim_value = sum(ssim_arr) / len(ssim_arr)
    entropy_enhanced = sum(entropy_enhanced_arr) / len(entropy_enhanced_arr)
    std_dev_enhanced = sum(std_dev_enhanced_arr) / len(std_dev_enhanced_arr)
    mse_value = sum(mse_arr) / len(mse_arr)
    overall_mean_absolute_difference = np.mean(mean_absolute_differences)

    result_final = f'\n\n\nFinal: psnr: {psnr_value}, ssim: {ssim_value}, entropy: {entropy_enhanced}, std: {std_dev_enhanced}, mse: {mse_value}'
    print(result_final)
    print(f"Overall Mean Absolute Difference: {overall_mean_absolute_difference:.5f}")

