import cv2
import os
import time
# Create a directory to store photos
output_dir = "new_real/test"
os.makedirs(output_dir, exist_ok=True)

# Prompt user for folder name and student ID
folder_name = input("Enter the folder name (e.g., real_dark): ")
# student_id = input("Enter the student ID: ")

# Create a subdirectory inside the main folder
folder_path = os.path.join(output_dir, folder_name)
os.makedirs(folder_path, exist_ok=True)
# subfolder_path = os.path.join(folder_path, student_id)
# os.makedirs(subfolder_path, exist_ok=True)
# # Initialize camera
camera = cv2.VideoCapture(0)

# Turn on the camera feed for 20 seconds
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = camera.read()
    cv2.imshow("Camera Feed", frame)
    cv2.waitKey(1)  # Display the live feed

# Capture 10 photos
for i in range(1, 6):
    ret, frame = camera.read()

    # Save the image with the specified filename
    filename = f"{folder_name}_{i}.jpg"
    image_path = os.path.join(folder_path, filename)
    cv2.imwrite(image_path, frame)

    # print(f"Saved photo {i} as {filename}")

# Release camera
camera.release()
cv2.destroyAllWindows()

# print(f"Photos captured and saved in folder '{folder_name}'.")
