import cv2
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import correlate
from scipy.stats import norm

def skew(x):
    mean = np.mean(x)
    std = np.std(x)
    return np.mean(((x - mean) / std) ** 3)

def kurtosis(x):
    mean = np.mean(x)
    std = np.std(x)
    return np.mean(((x - mean) / std) ** 4) - 3

def compute_mscn_coefficients(image, C=1.0/255):
    kernel = np.ones((7, 7)) / 49.0
    mu = correlate(image, kernel, mode='nearest')
    mu = np.nan_to_num(mu)  # Replace NaNs and Infs with finite numbers
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(correlate(image * image, kernel, mode='nearest') - mu_sq))
    sigma = np.nan_to_num(sigma)  # Replace NaNs and Infs with finite numbers
    structdis = (image - mu) / (sigma + C)
    structdis = np.nan_to_num(structdis)  # Replace NaNs and Infs with finite numbers
    return structdis

def extract_quality_features(image):
    mscn_coefficients = compute_mscn_coefficients(image)
    features = []

    block_size = 8
    height, width = mscn_coefficients.shape

    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            block = mscn_coefficients[i:i + block_size, j:j + block_size].ravel()
            mean = np.mean(block)
            std = np.std(block)
            # Check if standard deviation is close to zero to avoid numerical instability
            if std < 1e-6:
                skewness = 0  # Set skewness to zero (or any default value)
                kurt = 0      # Set kurtosis to zero (or any default value)
            else:
                skewness = skew(block)
                kurt = kurtosis(block)
            features.append([
                mean,
                std,
                skewness,
                kurt
            ])
    
    return np.array(features)

def compute_niqe(image):
    features = extract_quality_features(image)
    mu_distparam = np.mean(features, axis=0)
    cov_distparam = np.cov(features.T)

    mu_prisparam = np.array([0.1840, 0.1840, 0.1840, 0.1840])
    cov_prisparam = np.array([
        [0.0179, 0.0179, 0.0179, 0.0179],
        [0.0179, 0.0179, 0.0179, 0.0179],
        [0.0179, 0.0179, 0.0179, 0.0179],
        [0.0179, 0.0179, 0.0179, 0.0179]
    ])

    # Ensure that mu_distparam and mu_prisparam have the same shape
    if mu_distparam.shape != mu_prisparam.shape:
        raise ValueError(f"Shape mismatch: mu_distparam {mu_distparam.shape}, mu_prisparam {mu_prisparam.shape}")
    
    X = mu_distparam - mu_prisparam
    cov_mean = (cov_prisparam + cov_distparam) / 2

    # Check for valid covariance matrix
    if np.any(np.isnan(cov_mean)) or np.any(np.isinf(cov_mean)):
        raise ValueError("Covariance matrix contains NaNs or Infs")

    inv_cov_mean = np.linalg.inv(cov_mean)
    niqe_score = np.sqrt(np.dot(np.dot(X, inv_cov_mean), X))
    return niqe_score

def load_and_compute_niqe(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load image")

    # Normalize image to [0, 1] range
    image = image.astype(np.float32) / 255.0

    # Compute NIQE score
    niqe_score = compute_niqe(image)
    return niqe_score

# Example usage
if __name__ == "__main__":
    folder = 'Dataset/Zero-DCE++/result_test' 
    images = os.listdir(folder)
    niqe_arr = []
    for image in tqdm(images):
        image_path = os.path.join(folder,image)  # Replace with your image path
        try:
            niqe_score = load_and_compute_niqe(image_path)
            niqe_arr.append(niqe_score)
        except Exception as e:
            print(f'Error: {e}')
    niqe_score = sum(niqe_arr)/len(niqe_arr)
    print(f'NIQE score: {niqe_score:.5f}')

