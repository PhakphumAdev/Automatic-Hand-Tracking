# part 1
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import argparse
from sam2.build_sam import build_sam2_video_predictor
import os
import matplotlib.pyplot as plt
from PIL import Image

def detect_hands(input_image):
    """
    PART 1: Detect hands in an image using MediaPipe
    """
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(input_image)
    detection_result = detector.detect(image)

    # Load image and get dimensions
    image = mp.Image.create_from_file(input_image)
    image_width = image.width  # Get pixel width
    image_height = image.height  # Get pixel height

    # Convert normalized coordinates to pixel coordinates
    hand_landmarks = []
    for hand in detection_result.hand_landmarks:
        landmarks = np.array([[int(lm.x * image_width), int(lm.y * image_height)] for lm in hand])  # Ensure integer pixel coordinates
        hand_landmarks.append(landmarks)

    
    num_hands = len(detection_result.hand_landmarks)
    return num_hands, np.array(hand_landmarks)
def get_hand_boxes(hand_landmarks, image_size):
    boxes = []
    for hand in hand_landmarks:
        x_coords = hand[:, 0]
        y_coords = hand[:, 1]
        # Return boxes in [x1, y1, x2, y2] format
        box = [
            max(0, np.min(x_coords) - 15),  # x_min
            max(0, np.min(y_coords) - 15),  # y_min
            min(image_size[0], np.max(x_coords) + 15),  # x_max
            min(image_size[1], np.max(y_coords) + 15)   # y_max
        ]
        boxes.append(np.array(box, dtype=np.float32))  # Ensure float32
    return boxes
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
        frame_filename = os.path.join(output_dir, f"{frame_idx:04d}.jpg")  # Remove "frame_" prefix
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def track_hands(input_video_path, output_video_path):
    """
    PART 2: Track hands in a video and overlay masks using SAM 2
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video2image(input_video_path)
    sam2_checkpoint = "/home/pa2497/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_dir = os.path.join("video_as_image", video_name)
    video_dir = output_dir

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=video_dir)

    # Process frame 0
    frame_idx = 0
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    frame = cv2.imread(frame_path)

    # Detect hands and refine the annotations for frame 0
    num_hands, hand_landmarks = detect_hands(frame_path)  # hand_landmarks shape: (num_hands, 21, 2)

    # Get boxes instead of points
    image = mp.Image.create_from_file(frame_path)
    image_width = image.width  # Get pixel width
    image_height = image.height  # Get pixel height

    image_size = (image_height, image_width)
    boxes = get_hand_boxes(hand_landmarks, image_size)

    # Reshape to [N, 2] where N = num_hands * 21
    #points = hand_landmarks.reshape(-1, 2).astype(np.float32)  # Shape: (num_hands * 21, 2)
    #labels = np.ones(points.shape[0], dtype=np.int32)  # Shape: (num_hands * 21,)

    # (Optional) If you want per-hand interaction (e.g., track each hand separately):
    #prompts = {
    #    i: (hand.astype(np.float32), np.ones(21, dtype=np.int32)) 
    #    for i, hand in enumerate(hand_landmarks)
    #}

    # Pass all points and labels to SAM
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=num_hands,
        #labels=labels,
        box=boxes,
    )

    binary_mask = (out_mask_logits[0] > 0.5).cpu().numpy()
    processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask(processed_mask, plt.gca(),alpha=0.5)
    plt.savefig("output.jpg")  # JPEG
    plt.close()  # Free memory by closing the figure

    # Visualize the results
    #plt.figure(figsize=(9, 6))
    #plt.title(f"Frame {frame_idx}")
    #plt.imshow(Image.open(frame_path))
    #show_points(points, labels, plt.gca())
    #for i, out_obj_id in enumerate(out_obj_ids):
    #    show_points(*prompts[out_obj_id], plt.gca())
    #    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    #plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path')
    parser.add_argument('output_file_path')

    args = parser.parse_args()
    track_hands(args.input_file_path, args.output_file_path)

if __name__=="__main__":
    main()
