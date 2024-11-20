import cv2
import mediapipe as mp
import numpy as np
import subprocess

def process_video():
    # Step 1: Background separation
    cap = cv2.VideoCapture('3762907-uhd_3840_2160_25fps.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use XVID codec for temporary output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = 'temp_output.avi'
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(frame_rgb)
            mask = results.segmentation_mask
            condition = mask > 0.5
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            output_image = np.where(condition[..., None], frame, bg_image)
            out.write(output_image)

    cap.release()
    out.release()

    # Step 2: Convert to MP4 using FFmpeg
    subprocess.run([
        'ffmpeg', '-i', temp_output,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        'final_output.mp4'
    ])

process_video()
print("Processing complete. Check final_output.mp4")