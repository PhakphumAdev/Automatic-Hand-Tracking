import cv2
import os

# Define the input directory and output video file
input_dir = "video_as_image/test/masked_frames"
output_video = "output_video.mp4"

# Get list of image frames sorted in order
frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

# Read the first frame to get dimensions
first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
height, width, _ = first_frame.shape

# Define video writer object (MP4V codec for MP4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' or 'MJPG' for AVI
fps = 1

out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames into the video
for frame_name in frames:
    frame_path = os.path.join(input_dir, frame_name)
    frame = cv2.imread(frame_path)
    out.write(frame)

# Release the video writer
out.release()

print(f"Video saved as {output_video}")