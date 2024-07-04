import random
import shutil
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from src.anti_spoof_predict import Detection
import os
import cv2
from src.anti_spoof_predict import AntiSpoofPredict, Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name
from function_model.SCI import LowLightEnhancer

def get_confidence(image):
    face_detection = Detection()
    image_bbox, conf = face_detection.get_bbox(image)
    return conf

def get_threshold(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_intensity = cv2.mean(gray_image)[0]
        return average_intensity

def apply_fft_and_remove_noise(image):
   blurred_img = cv2.fastNlMeansDenoisingColored(image, None, 3,3,7,21)
   return blurred_img

def crop_image(image, model_name, model_test, image_cropper):
    image_bbox, _ = model_test.get_bbox(image)
    if image_bbox is not None:
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
    return img

def calculate_crop_image():
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    model_1 = '2.7_80x80_MiniFASNetV2.pth'
    model_2 = '4_0_0_80x80_MiniFASNetV2.pth'
    model_llie = 'miniFAS/model_onnx/SCI.onnx'
    dataset_path = 'miniFAS/datasets/Test/new_dataset/train'
    dataset_train = 'miniFAS/datasets/Train/No_Noise'
    lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
    model1_name = '2.7_80x80'
    model2_name = '4_80x80'
    train_model1 = os.path.join(dataset_train, model1_name)
    train_model2 = os.path.join(dataset_train, model2_name)
    labels = os.listdir(dataset_path)
    for label in tqdm(labels):
        model1_path = os.path.join(train_model1, label)
        model2_path = os.path.join(train_model2, label)
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)
        for image in tqdm(images):
            img_path = os.path.join(label_path, image)
            img = cv2.imread(img_path)
            img = apply_fft_and_remove_noise(img)
            img = lowlight_enhancer.enhance(img) 
            conf = get_confidence(img)
            image = 're_noise_' + image
            if conf >= 0.9:
                img_1 = crop_image(img, model_1, model_test, image_cropper)
                img_2 = crop_image(img, model_2, model_test, image_cropper)
                cv2.imwrite(os.path.join(model1_path,image), img_1)
                cv2.imwrite(os.path.join(model2_path,image), img_2)

def shuffle_and_move_data():
    input_folder = "miniFAS/datasets/Test/rec_Hiep"
    folder_A = "miniFAS/datasets/Test/new_dataset/test/1"
    folder_B = "miniFAS/datasets/Test/new_dataset/train/1"
    num_images_A = 732
    images = os.listdir(input_folder)
    random.shuffle(images)

    os.makedirs(folder_A, exist_ok=True)
    os.makedirs(folder_B, exist_ok=True)

    for i, image in enumerate(images):
        if i < num_images_A:
            shutil.move(os.path.join(input_folder, image), os.path.join(folder_A, image))
        else:
            shutil.move(os.path.join(input_folder, image), os.path.join(folder_B, image))

def shuffle_and_move_folder():
    source_parent_dir = "miniFAS/datasets/real_extract_test"
    destination_dir = "miniFAS/datasets/randomly_5_folders"
    num_folders_to_move = 5
    all_folders = [folder for folder in os.listdir(source_parent_dir) if os.path.isdir(os.path.join(source_parent_dir, folder))]
    folders_to_move = random.sample(all_folders, min(num_folders_to_move, len(all_folders)))

    for folder in folders_to_move:
        source_path = os.path.join(source_parent_dir, folder)
        destination_path = os.path.join(destination_dir, folder)
        shutil.move(source_path, destination_path)
    print(f"{len(folders_to_move)} folders moved to {destination_dir}.")

def copy_files_to_new_folder():
    source_directory = "miniFAS/datasets/real_extract_test"
    destination_directory = "miniFAS/datasets/Test/new_dataset/test/1"
    sub_folder = os.listdir(source_directory)
    for subfold in sub_folder:
        new_name = '1_'+ str(subfold)
        sub_folder_path = os.path.join(source_directory, subfold)
        files = os.listdir(sub_folder_path)
        cnt = 0
        for file in files:
            cnt += 1
            new_name_file = new_name + '_' +str(cnt) + '.jpg'
            destination_file_path = os.path.join(destination_directory, new_name_file)
            source_file_path = os.path.join(sub_folder_path, file)
            shutil.copy2(source_file_path, destination_file_path)
            print(new_name_file)


if __name__ == "__main__":
   
    calculate_crop_image()
    # shuffle_and_move_data()
    # shuffle_and_move_folder()
    # copy_files_to_new_folder()
