# part 1
import cv2
import numpy as np
#import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
#from sam2 import SamVideoPredictor  # Assuming this is the correct import
#import torch
import argparse
#from sam2.build_sam import build_sam2_video_predictor
import os

def detect_hands(image):
    """
    PART 1: Detect hands in an image using MediaPipe
    """
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    image = mp.Image.create_from_file("image.jpg")
    detection_result = detector.detect(image)

    hand_landmarks = []
    for hand in detection_result.hand_landmarks:
    # Convert each hand's landmarks to a NumPy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand])
        hand_landmarks.append(landmarks)
    
    num_hands = len(detection_result.hand_landmarks)
    return num_hands, np.array(hand_landmarks)
def video2image(video_path):
    """
    Convert video to image frames and save them in a folder named after the video file
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # Create directory to save frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("video_as_image", video_name)
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    
def track_hands(input_video_path, output_video_path):
    """
    PART 2: Track hands in a video and overlay masks using SAM 2
    """
    video2image(input_video_path)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path')
    parser.add_argument('output_file_path')

    args = parser.parse_args()
    track_hands(args.input_file_path, args.output_file_path)

if __name__=="__main__":
    main()
