import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame interval for 0.5 second
    frame_interval = int(fps * 100)
    
    # Initialize variables
    frame_count = 0
    success = True
    
    # Loop through the video frames
    while success:
        # Set the frame position to the next 0.5 second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        
        # Read the next frame
        success, image = cap.read()
        
        # If a frame is read successfully, save it
        if success:
            # Save the frame
            frame_path = os.path.join(output_folder, f"frame_{frame_count + 1}.jpg")
            cv2.imwrite(frame_path, image)
            
            # Increment frame count
            frame_count += 1
    
    # Release the video capture object
    cap.release()


if __name__ == "__main__":
    # Get video path and output folder name from user input
    video_path = input("Enter the path to the video file: ")
    output_fold = input("Enter the name of the output folder to save frames: ")
    output_folder = os.path.join('real_extract_test',output_fold)
    # Extract frames
    extract_frames(video_path, output_folder)
    
    print("Frames extraction complete!")
