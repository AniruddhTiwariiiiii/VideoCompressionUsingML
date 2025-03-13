#-------------------ALL IMPORTANT LBRARIES------------------
import cv2
import os
import os
import cv2
import numpy as np

!apt-get install ffmpeg
!pip install ffmpeg-python
!apt-get update
!apt-get install ffmpeg
import ffmpeg
import os
pip install scenedetect
!pip install pydub
!sudo apt install ffmpeg
from pydub import AudioSegment


#------FRAME EXTRACTION----------------------------------
def extract_frames(video_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    vidcap = cv2.VideoCapture(video_path)


    if not vidcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return


    count = 0
    while True:
        success, frame = vidcap.read()


        if not success:
            break

        # Save the current frame as an image file (e.g., .jpg)
        frame_filename = os.path.join(output_folder, f"frame{count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        count += 1

    # Release the video capture object
    vidcap.release()
    print("Frame extraction completed.")

# Usage example
video_path = 'AlitaBattleAngel.mkv'  # Your input .mkv video file
output_folder = 'EXTRACTED FRAMES'        # Directory to save extracted frames
num_frames=extract_frames(video_path,output_folder)

#---------FOLDER SIZE OF EXTRACTED FRAMES-----------

def get_folder_size(folder_path):
    """Returns the size of the folder in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            # Skip if the file is not accessible
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

# Example usage
folder_path = 'EXTRACTED FRAMES'  # Change this to your folder path
folder_size_bytes = get_folder_size(folder_path)
print(f"Folder size: {folder_size_bytes / (1024**3):.2f} GB")


#----------Number of images in folder--------------------

def count_images_in_folder(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Filter files based on the extensions (i.e., images)
    image_files = [file for file in files if os.path.splitext(file)[1].lower() in extensions]

    # Return the number of image files
    return len(image_files)

# Usage example
folder_path = 'EXTRACTED FRAMES'  # Update with your folder path
num_images = count_images_in_folder(folder_path)
print(f"Number of images in the folder: {num_images}")


#----------PYSCENE DETECT-----------------------
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

video_manager = VideoManager(['AlitaBattleAngel.mkv'])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
scenes = scene_manager.get_scene_list()
print(scenes)  # List of scenes with start/end times
len(scenes)

# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# import cv2
# import os

# Initialize VideoManager and SceneManager.
video_manager = VideoManager(['AlitaBattleAngel.mkv'])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))

# Start video_manager to process scenes.
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)

# Retrieve list of scenes.
scenes = scene_manager.get_scene_list()
print(f"Detected {len(scenes)} scenes.")

# Set up to save frames.
output_dir = "scene_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file using OpenCV.
video_path = 'AlitaBattleAngel.mkv'  # Define video_path here
cap = cv2.VideoCapture(video_path)  # Initialize 'cap' here

# Iterate through each scene to capture the start frame.
for i, scene in enumerate(scenes):
    # Get the start frame number of the scene.
    start_frame = scene[0].get_frames()

    # Set the video to the starting frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret_val, frame = cap.read()  # Use cap.read() to get the frame

    # Check if frame was successfully captured.
    if ret_val:
        # Save frame as an image file with the correct path format.
        frame_path = os.path.join(output_dir, f"scene_{i + 1}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame for scene {i + 1} at {frame_path}")

# Release resources.
cap.release()  # Release the OpenCV video capture object
video_manager.release()
print("Frame extraction completed.")


#----KEYFRAME SELECTION USING SBD --------------------
#HISTOGRAM
'''

# Directory with saved scene frames.
scene_dir = "scene_frames"
key_frame_output_dir = "KEYFRAMES_HIST"
os.makedirs(key_frame_output_dir, exist_ok=True)

# Threshold for histogram difference to detect key frames.
hist_diff_threshold = 0.6  # Adjust this based on sensitivity

# Function to compute histogram and normalize.
def compute_histogram(frame):
    # Convert frame to grayscale for histogram calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

# Process each scene file.
for scene_file in sorted(os.listdir(scene_dir)):
    # Define the path for each scene and open it with OpenCV.
    scene_path = os.path.join(scene_dir, scene_file)
    cap = cv2.VideoCapture(scene_path)

    # Initialize variables to hold the previous frame and histogram.
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Could not read {scene_file}")
        cap.release()
        continue

    # Compute the histogram of the first frame.
    prev_hist = compute_histogram(prev_frame)
    key_frame_count = 0

    # Loop over frames in the current scene.
    frame_index = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Compute histogram for the current frame.
        curr_hist = compute_histogram(curr_frame)

        # Calculate the histogram difference (correlation).
        hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

        # If histogram difference is below the threshold, it's a key frame.
        if hist_diff < hist_diff_threshold:
            key_frame_count += 1
            key_frame_path = os.path.join(key_frame_output_dir, f"{scene_file}_keyframe_{key_frame_count}.jpg")
            cv2.imwrite(key_frame_path, curr_frame)
            print(f"Saved key frame {key_frame_count} for {scene_file} at {key_frame_path}")

            # Update prev_hist to avoid saving similar frames
            prev_hist = curr_hist

        frame_index += 1

    cap.release()

print("Key frame extraction completed.")
'''

#-----CALCULATING HISTOGRAM DIFFERENCES------------------

# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# import cv2
# import os
# import numpy as np

# # Initialize VideoManager and SceneManager.
# video_manager = VideoManager(['AlitaBattleAngel.mkv'])
# scene_manager = SceneManager()
# scene_manager.add_detector(ContentDetector(threshold=30.0))

# # Start video_manager to process scenes.
# video_manager.start()
# scene_manager.detect_scenes(frame_source=video_manager)

# # Retrieve list of scenes.
# scenes = scene_manager.get_scene_list()
# print(f"Detected {len(scenes)} scenes.")

# # Set up to save keyframes.
# keyframe_dir = "key_frames"
# os.makedirs(keyframe_dir, exist_ok=True)

# # Open the video file using OpenCV.
# video_path = 'AlitaBattleAngel.mkv'
# cap = cv2.VideoCapture(video_path)

# # Histogram threshold for keyframe selection (adjustable based on testing).
# hist_diff_threshold = 3000
# total_keyframes = 0

# # Iterate over each scene to capture keyframes.
# for i, scene in enumerate(scenes):
#     start_frame = scene[0].get_frames()
#     end_frame = scene[1].get_frames()

#     # Initialize previous histogram for comparison.
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     _, prev_frame = cap.read()
#     prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

#     # Loop through frames in the scene to detect keyframes.
#     for frame_num in range(start_frame + 1, end_frame):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret_val, frame = cap.read()
#         if not ret_val:
#             break

#         # Calculate histogram difference.
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
#         hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

#         # Save frame if histogram difference exceeds threshold.
#         if hist_diff > hist_diff_threshold:
#             keyframe_path = os.path.join(keyframe_dir, f"keyframe_scene_{i + 1}_frame_{frame_num}.jpg")
#             cv2.imwrite(keyframe_path, frame)
#             total_keyframes += 1
#             prev_hist = hist  # Update previous histogram for the next frame.
#             print(f"Saved keyframe for scene {i + 1} at frame {frame_num}, hist_diff: {hist_diff}")

#         # Stop if the total keyframes exceed the target range.
#         if total_keyframes >= 1500:
#             break

#     # Stop if the total keyframes exceed the target range.
#     if total_keyframes >= 1500:
#         break

# # Release resources.
# cap.release()
# video_manager.release()
# print(f"Keyframe extraction completed with {total_keyframes} keyframes saved.")


#---------Interval Based Selection------------------

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os
import numpy as np

# Initialize VideoManager and SceneManager.
video_manager = VideoManager(['/content/drive/MyDrive/PROJECT_VC(SELF)/AlitaBattleAngel.mkv'])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))

# Start video_manager to process scenes.
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)

# Retrieve list of scenes.
scenes = scene_manager.get_scene_list()
print(f"Detected {len(scenes)} scenes.")

# Set up to save keyframes.
keyframe_dir = "./keyframe_interval_based_selection"
os.makedirs(keyframe_dir, exist_ok=True)

# Open the video file using OpenCV.
video_path = './AlitaBattleAngel.mkv'
cap = cv2.VideoCapture(video_path)

# Fixed number of keyframes per scene (e.g., 5-6)
keyframes_per_scene = 6
total_keyframes = 0

# Iterate over each scene to capture keyframes.
for i, scene in enumerate(scenes):
    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()

    # Number of frames in the scene
    num_frames_in_scene = end_frame - start_frame

    # Calculate interval between keyframes
    if num_frames_in_scene > keyframes_per_scene:
        interval = num_frames_in_scene // keyframes_per_scene
    else:
        interval = 1  # If there are fewer frames, select one keyframe per frame

    # Initialize the frame counter
    frame_counter = start_frame

    # Loop through frames in the scene to select keyframes
    for j in range(keyframes_per_scene):
        # Ensure the selected frame is within the scene
        if frame_counter >= end_frame:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # Save the keyframe
        keyframe_path = os.path.join(keyframe_dir, f"keyframe_scene_{i + 1}frame{frame_counter}.jpg")
        cv2.imwrite(keyframe_path, frame)
        total_keyframes += 1
        print(f"Saved keyframe for scene {i + 1} at frame {frame_counter}")

        # Move to the next frame interval
        frame_counter += interval

        # Stop if the total keyframes exceed the target range.
        if total_keyframes >= 1500:
            break

    # Stop if the total keyframes exceed the target range.
    if total_keyframes >= 1500:
        break

# Release resources.
cap.release()
video_manager.release()
print(f"Keyframe extraction completed with {total_keyframes} keyframes saved.")

#---------DCT FRAMES COMP--------------------


def apply_dct_on_block(block):
    """Apply DCT on an 8x8 block."""
    return cv2.dct(block.astype(np.float32))

def quantize_dct_block(dct_block, quantization_matrix):
    """Quantize the DCT coefficients in the block."""
    return np.round(dct_block / quantization_matrix)

def dequantize_dct_block(dct_block, quantization_matrix):
    """Dequantize the DCT coefficients in the block."""
    return dct_block * quantization_matrix

def apply_idct_on_block(dct_block):
    """Apply IDCT on an 8x8 block."""
    return cv2.idct(dct_block).clip(0, 255)

def compress_image_dct(image, quantization_matrix):
    """Compress a color image using DCT and quantization."""
    height, width, channels = image.shape
    compressed_image = np.zeros((height, width, channels), dtype=np.float32)
    
    # Process each channel separately
    for c in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = image[i:i+8, j:j+8, c]
                dct_block = apply_dct_on_block(block)
                quantized_block = quantize_dct_block(dct_block, quantization_matrix)
                compressed_image[i:i+8, j:j+8, c] = quantized_block

    return compressed_image

def decompress_image_dct(compressed_image, quantization_matrix):
    """Decompress a color image using IDCT and dequantization."""
    height, width, channels = compressed_image.shape
    decompressed_image = np.zeros((height, width, channels), dtype=np.float32)

    # Process each channel separately
    for c in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = compressed_image[i:i+8, j:j+8, c]
                dequantized_block = dequantize_dct_block(block, quantization_matrix)
                idct_block = apply_idct_on_block(dequantized_block)
                decompressed_image[i:i+8, j:j+8, c] = idct_block

    return decompressed_image.astype(np.uint8)

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and decompressed image."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # No error
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mae(original, reconstructed):
    """Calculate MAE between original and decompressed image."""
    return np.mean(np.abs(original - reconstructed))

def calculate_compression_ratio(original, compressed):
    """Calculate the compression ratio."""
    original_size = original.nbytes
    compressed_size = compressed.nbytes
    return original_size / compressed_size

def process_keyframes_dct(input_folder, output_folder, output_folder1, quantization_matrix):
    """Apply DCT compression on color keyframes and save compressed images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

    filenames = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]
    
    for filename in filenames:
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)  # Read color image

        # Compress and decompress the image
        compressed_image = compress_image_dct(image, quantization_matrix)
        decompressed_image = decompress_image_dct(compressed_image, quantization_matrix)

        # Save the compressed image to Folder3 (quantized DCT coefficients)
        compressed_output_path = os.path.join(output_folder, filename)
        cv2.imwrite(compressed_output_path, compressed_image.clip(0, 255).astype(np.uint8))

        # Save the decompressed image to Folder4
        decompressed_output_path = os.path.join(output_folder1, filename)
        cv2.imwrite(decompressed_output_path, decompressed_image)

        # Calculate performance metrics: MAE, PSNR, and Compression Ratio
        psnr = calculate_psnr(image, decompressed_image)
        mae = calculate_mae(image, decompressed_image)
        compression_ratio = calculate_compression_ratio(image, compressed_image)

        print(f"Metrics for {filename}:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  MAE: {mae:.2f}")
        print(f"  Compression Ratio: {compression_ratio:.2f}")
        print('-' * 40)

# Define a standard quantization matrix (similar to JPEG)
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Usage example
input_folder = 'keyframe_interval_based_selection'  # Folder with keyframes
output_folder = 'c_keyframe'  # Folder to save compressed images
output_folder1 = 'dc_keyframe'  # Folder to save decompressed images
process_keyframes_dct(input_folder, output_folder, output_folder1, quantization_matrix)

#CHANGE IF REQUIRED ANY

#---------------COMPRESSION TECHNIQUE------------
def load_keyframe_timestamps(csv_file):
    """
    Load the keyframe filenames and their corresponding timestamps from the CSV file.
    """
    keyframe_data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            frame_name = row[0]
            timestamp = float(row[1]) if row[1] else None
            if frame_name and timestamp is not None:
                keyframe_data.append((frame_name, timestamp))
    return keyframe_data

def filter_optical_flow(flow, threshold=1.0):
    """
    Refines the optical flow by removing outliers (e.g., vectors with very large magnitudes)
    that could cause artifacts in interpolation.
    """
    flow_magnitude = np.linalg.norm(flow, axis=-1)
    flow_filtered = flow * (flow_magnitude[..., None] > threshold)
    return flow_filtered

def warp_frame(frame, flow):
    """
    Warp a frame using the motion vectors.
    """
    h, w = flow.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    flow_map += flow
    flow_map[..., 0] = np.clip(flow_map[..., 0], 0, w - 1)
    flow_map[..., 1] = np.clip(flow_map[..., 1], 0, h - 1)
    return cv2.remap(frame, flow_map, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def calculate_mse(frame1, frame2):
    """
    Calculate Mean Squared Error between two frames.
    """
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)
    return np.mean(np.square(diff))

def dense_optical_flow(prev_gray, current_gray):
    """
    Compute dense optical flow using Farneback's method.
    """
    return cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#----------------------DECOMPRESSION TECHNIQUE---------------------------

def interpolate_with_optical_flow(prev_keyframe, current_keyframe, flow, alpha):
    """
    Interpolate between two frames using optical flow.
    """
    warped_prev_frame = warp_frame(prev_keyframe, flow * (1 - alpha))
    warped_current_frame = warp_frame(current_keyframe, flow * alpha)
    interpolated_frame = (1 - alpha) * warped_prev_frame + alpha * warped_current_frame
    return np.clip(interpolated_frame, 0, 255).astype(np.uint8)

def process_frames_and_create_video(keyframe_folder, video_folder, video_filename, csv_file, output_fps=23.80952380952381, mse_threshold=500, artifact_threshold=500):
    """
    Process keyframes, interpolate intermediate frames using motion estimation, and create a video.
    """
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    keyframe_data = load_keyframe_timestamps(csv_file)
    keyframe_data.sort(key=lambda x: x[1])  # Sort keyframes by timestamp
    keyframe_filenames = [frame[0] for frame in keyframe_data]
    timestamps = [frame[1] for frame in keyframe_data]

    first_keyframe_path = os.path.join(keyframe_folder, keyframe_filenames[0])
    first_keyframe = cv2.imread(first_keyframe_path)
    height, width, channels = first_keyframe.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, output_fps, (width, height))

    frame_count = 0
    prev_intermediate_frame = None

    for i in range(1, len(keyframe_filenames)):
        prev_keyframe_path = os.path.join(keyframe_folder, keyframe_filenames[i - 1])
        current_keyframe_path = os.path.join(keyframe_folder, keyframe_filenames[i])

        prev_keyframe = cv2.imread(prev_keyframe_path)
        current_keyframe = cv2.imread(current_keyframe_path)

        prev_timestamp = timestamps[i - 1]
        current_timestamp = timestamps[i]
        time_diff = current_timestamp - prev_timestamp
        num_intermediate_frames = int(output_fps * time_diff) - 1

        mse = calculate_mse(prev_keyframe, current_keyframe)
        prev_gray = cv2.cvtColor(prev_keyframe, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_keyframe, cv2.COLOR_BGR2GRAY)
        flow = dense_optical_flow(prev_gray, current_gray)
        filtered_flow = filter_optical_flow(flow)

        if mse > mse_threshold or np.max(np.abs(filtered_flow)) > artifact_threshold:
            print(f"Frames {i-1} and {i} are significantly different, duplicating the keyframe")
            for j in range(1, num_intermediate_frames + 1):
                dusky_frame = prev_keyframe
                out.write(dusky_frame)
                cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count:04d}.jpg"), dusky_frame)
                frame_count += 1
        else:
            print(f"Frames {i-1} and {i} are similar, using motion estimation")
            alpha_step = 1 / (num_intermediate_frames + 1)
            for j in range(1, num_intermediate_frames + 1):
                alpha = j * alpha_step
                interpolated_frame = interpolate_with_optical_flow(prev_keyframe, current_keyframe, filtered_flow, alpha)
                if prev_intermediate_frame is not None:
                    prev_mse = calculate_mse(prev_intermediate_frame, interpolated_frame)
                    if prev_mse > artifact_threshold:
                        print(f"Artifact detected, but retaining the frame.")
                prev_intermediate_frame = interpolated_frame
                out.write(interpolated_frame)
                cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count:04d}.jpg"), interpolated_frame)
                frame_count += 1

    out.release()
    print(f"Video created successfully at {video_filename}")
    print(f"Frames saved to folder: {video_folder}")


#------------#REGENERATING VIDEO-------------------

keyframe_folder = "dc_keyframe"
video_folder = "recombined_vid_frames"
video_filename = "recombined_vid6.avi"
csv_file = "keyframe_details (1).csv"

process_frames_and_create_video(keyframe_folder, video_folder, video_filename, csv_file)



#----AUDIO--------------------------------------

#EXTRACTING AUDIO



# Define input video and output audio paths
input_video = 'AlitaBattleAngel.mkv'
output_audio = 'original_audio.aac'

# Mount Google Drive to ensure file access permissions
# This step is important for accessing files in Google Drive from within Colab
'''
from google.colab import drive
drive.mount('/content/drive')
'''


# Extract audio in AAC format
try:
    # Use os.path.abspath to get the absolute path of the input video
    # This ensures the path is correctly interpreted by ffmpeg
    absolute_input_path = os.path.abspath(input_video)

    (
        ffmpeg
        .input(absolute_input_path)
        .output(output_audio, **{'c:a': 'aac', 'b:a': '192k', 'map': 'a'})
        .run(capture_stdout=True, capture_stderr=True) # Capture stdout and stderr for debugging
    )
    print(f"Audio extracted and saved to: {output_audio}")
except ffmpeg.Error as e:
    print(f"An error occurred: {e.stderr.decode()}")
    # Print stderr for more detailed error information



#------------COVERT .aac to .mp3 SUPPORTING WITH GENERATED VIDEO-------------------
output_audio = 'original_audio.mp3'
ffmpeg.input(input_video).output(output_audio, **{'q:a': 0, 'map': 'a', 'acodec': 'mp3'}).run()

#---MERGE VIDEO WITH AUDIO------------

!ffmpeg -i "recombined_vid6.avi" -i "original_audio.mp3" -c:v copy -c:a aac -b:a 192k "final_vid.mp4"





##-----------------EVALUATION METRIC CALCULATION--------------------------




#-------#AUDIO EXTRACTIONS WITH DECOMRESSED VIDEO------------


# Define input video and output audio paths
input_video = 'final_vid.mp4'
output_audio = 'decompressed_vid__audio.aac'


# Mount Google Drive to ensure file access permissions
# This step is important for accessing files in Google Drive from within Colab
'''
from google.colab import drive
drive.mount('/content/drive')
'''


# Extract audio in AAC format
try:
    # Use os.path.abspath to get the absolute path of the input video
    # This ensures the path is correctly interpreted by ffmpeg
    absolute_input_path = os.path.abspath(input_video)

    (
        ffmpeg
        .input(absolute_input_path)
        .output(output_audio, **{'c:a': 'aac', 'b:a': '192k', 'map': 'a'})
        .run(capture_stdout=True, capture_stderr=True) # Capture stdout and stderr for debugging
    )
    print(f"Audio extracted and saved to: {output_audio}")
except ffmpeg.Error as e:
    print(f"An error occurred: {e.stderr.decode()}")
    # Print stderr for more detailed error information


#-----------CONVERT .AAC TO .WAV USING PYDUB-------------



# Load the .aac file
audio = AudioSegment.from_file("original_audio.aac", format="aac")

# Export as .wav
audio.export("original_audio.wav", format="wav")




#----MAE AND PSNR COMPUTATIONS------------------

import numpy as np
import librosa  # For audio processing
import cv2  # For video processing

def calculate_mae(original, decompressed):
    """
    Calculate Mean Absolute Error (MAE) between two signals (audio/video).
    """
    return np.mean(np.abs(original - decompressed))

def calculate_psnr(original, decompressed, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two signals (audio/video).
    Default max_value for audio is 1 (normalized between 0 and 1).
    """
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return float('inf')  # No error, identical signals
    return 10 * np.log10((max_value ** 2) / mse)

# For Audio comparison
def evaluate_audio(original_audio_path, decompressed_audio_path):
    # Load audio files using librosa
    original_audio, sr = librosa.load(original_audio_path, sr=None)
    decompressed_audio, _ = librosa.load(decompressed_audio_path, sr=sr)

    # Ensure both signals have the same length
    min_len = min(len(original_audio), len(decompressed_audio))
    original_audio = original_audio[:min_len]
    decompressed_audio = decompressed_audio[:min_len]

    # Calculate MAE and PSNR
    mae = calculate_mae(original_audio, decompressed_audio)
    psnr = calculate_psnr(original_audio, decompressed_audio)

    return mae, psnr

# For Video comparison (if you're dealing with video frames)
def evaluate_video(original_video_path, decompressed_video_path):
    # Read the videos using OpenCV
    original_video = cv2.VideoCapture(original_video_path)
    decompressed_video = cv2.VideoCapture(decompressed_video_path)

    frame_count = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))
    mae_values = []
    psnr_values = []

    for frame_num in range(frame_count):
        ret1, original_frame = original_video.read()
        ret2, decompressed_frame = decompressed_video.read()

        if not ret1 or not ret2:
            break

        # Convert frames to grayscale if they are color images
        original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        decompressed_frame_gray = cv2.cvtColor(decompressed_frame, cv2.COLOR_BGR2GRAY)

        # Calculate MAE and PSNR for the current frame
        mae = calculate_mae(original_frame_gray, decompressed_frame_gray)
        psnr = calculate_psnr(original_frame_gray, decompressed_frame_gray, max_value=255)

        mae_values.append(mae)
        psnr_values.append(psnr)

    # Calculate average MAE and PSNR over all frames
    avg_mae = np.mean(mae_values)
    avg_psnr = np.mean(psnr_values)

    original_video.release()
    decompressed_video.release()

    return avg_mae, avg_psnr


# Example Usage for Audio
original_audio_path = 'original_audio.wav'
decompressed_audio_path = 'decom_audio.wav'
mae_audio, psnr_audio = evaluate_audio(original_audio_path, decompressed_audio_path)
print(f"Audio MAE: {mae_audio:.4f}, Audio PSNR: {psnr_audio:.4f}")

# Example Usage for Video
original_video_path = 'AlitaBattleAngel.mkv'
decompressed_video_path = 'final_vid.mp4'
mae_video, psnr_video = evaluate_video(original_video_path, decompressed_video_path)
print(f"Video MAE: {mae_video:.4f}, Video PSNR: {psnr_video:.4f}")




#----COPUTING COMPRESSION RATIO-------------------
import os

original_size = os.path.getsize("AlitaBattleAngel.mkv")  # Size in bytes
compressed_size = os.path.getsize("final_vid.mp4")

compression_ratio = original_size / compressed_size
print(f"Compression Ratio: {compression_ratio:.2f}")



#-------------COMPUTE AVERAGE EUCLIDEAN DISTANCE-----------

import cv2
import numpy as np

# Load original and decompressed videos
original_video = cv2.VideoCapture("AlitaBattleAngel.mkv")
decompressed_video = cv2.VideoCapture("final_vid.mp4")

total_euclidean_distance = 0
frame_count = 0

while True:
    ret1, original_frame = original_video.read()
    ret2, decompressed_frame = decompressed_video.read()

    # Break the loop if we reach the end of either video
    if not ret1 or not ret2:
        break

    # Convert frames to grayscale for simplicity
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    decompressed_gray = cv2.cvtColor(decompressed_frame, cv2.COLOR_BGR2GRAY)

    # Compute Euclidean distance for the current frame
    distance = np.linalg.norm(original_gray - decompressed_gray)
    total_euclidean_distance += distance
    frame_count += 1

# Average Euclidean distance over all frames
average_euclidean_distance = total_euclidean_distance / frame_count
print(f"Average Euclidean Distance: {average_euclidean_distance:.2f}")
