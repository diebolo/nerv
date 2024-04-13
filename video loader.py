import cv2
import os

def extract_frames(video_path, start_frame, end_frame, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Set the starting frame (I hate opencv)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Loop through the frames within the specified range
    for frame_number in range(start_frame, end_frame):
        # Read the current frame
        ret, frame = video.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as a PNG file
        digit = frame_number - (start_frame - 1) # make the first frame 1
        output_file = os.path.join(output_folder, f'f{digit:05d}.png')
        cv2.imwrite(output_file, frame)

    # Release the video file
    video.release()

def reconstruct_video(input_folder, output_path, fps=24):
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    image_files.sort()

    # Get the first image to determine the frame size
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))    
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through the image files and write them to the video

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        video.write(image)

    # Release the video writer
    video.release()


# Extract frames from video at video path
video_path = 'dune trailer.mp4'
start_frame = 816 - 24
end_frame = 840
output_folder = 'data/dune'

# extract_frames(video_path, start_frame, end_frame, output_folder)

# Reconstruct video from prediction images
input_folder = 'output\honeybee\honeybee\embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low96_blk1_cycle1_gap1_e1200_warm240_b1_conv_lr0.0005_cosine_Fusion6_Strd5,3,2,2,2_SinRes_eval_actswish_\Visualize'
output_path = 'test.mp4'
reconstruct_video(input_folder, output_path)