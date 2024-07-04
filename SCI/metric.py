import os
import cv2
import numpy as np
import torch
from scipy.stats import entropy
from matplotlib import pyplot as plt


def calculate_metrics(enhanced_img,enhanced_img_gray):
    # Chuyển đổi ảnh từ [0, 255] sang [0, 1]
    enhanced_img_gray = enhanced_img_gray.astype(np.float32) / 255.0

    # Entropy
    entropy_enhanced = entropy(enhanced_img_gray.flatten())

    # Standard Deviation
    std_dev_enhanced = np.std(enhanced_img_gray)

    # Histogram 
    # Chuyển đổi ảnh sang không gian màu BGR
    num_bins = 256
    b, g, r = cv2.split(enhanced_img)

    # Tính histogram cho từng kênh màu
    hist_b = cv2.calcHist([b], [0], None, [num_bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [num_bins], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [num_bins], [0, 256])

    return (
        entropy_enhanced,
        std_dev_enhanced,
        hist_b,
        hist_g,
        hist_r,
    )


if __name__ == '__main__':
    enhanced_image_folder = (
        '/home/linhhima/enhanced_image_Linh/SCI/results/difficult'
    )

    images = os.listdir(enhanced_image_folder)
    results = ''
    sum_entropy = 0
    sum_std_dev = 0
    # Khởi tạo các biến để tính tổng histogram
    total_hist = np.zeros((3, 256))
    for image in images:
        enhanced_image_path = os.path.join(enhanced_image_folder, image)
        enhanced_img_gray = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(enhanced_image_path)
        (
            entropy_enhanced,
            std_dev_enhanced,
            hist_b,
            hist_g,
            hist_r,
        ) = calculate_metrics(enhanced_img, enhanced_img_gray)
        sum_entropy += entropy_enhanced
        sum_std_dev += std_dev_enhanced
        plt.plot(hist_b,color = 'b')
        plt.plot(hist_g,color = 'g')
        plt.plot(hist_r,color = 'r')
        # Lưu vào thư mục
        plt.savefig('foo.png')
        plt.clf()
        # Thêm histogram của ảnh vào tổng histogram
        # total_hist[0] += hist_b.flatten()
        # total_hist[1] += hist_g.flatten()
        # total_hist[2] += hist_r.flatten()
        result = f'{image} entropy: {entropy_enhanced}, std: {std_dev_enhanced}'
        results = results + result + '\n'
    average_entropy = sum_entropy / len(images)
    average_std_dev = sum_std_dev / len(images)
    # total_hist /= len(images)
    # plt.plot(total_hist[0],color = 'b')
    # plt.plot(total_hist[1],color = 'g')
    # plt.plot(total_hist[2],color = 'r')
    # plt.xlim([0,256])
    # plt.show()
    results = results + f'average entropy: {average_entropy}, std: {average_std_dev}'
    print(results)
