import cv2
import os

def record_video(output_folder, video_name):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(output_folder, f'{video_name}.avi'), fourcc, 20.0, (640, 480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Write the frame
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        else:
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get the folder name from user input
    folder_name = input("Enter the folder name: ")
    output_folder = os.path.join('video_real', folder_name)
    
    # Record video
    record_video(output_folder, 'recorded_video')
