import cv2
import mediapipe as mp
import numpy as np
import subprocess


def generate_aurora_pattern(width, height, shift):
    """오로라 효과 생성 (Perlin-like Noise 기반)"""
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Shift 값으로 움직임 효과 추가
    pattern = np.sin(x_grid + shift) + np.cos(y_grid + shift / 2)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())  # Normalize

    # Apply color gradients for aurora
    aurora = np.zeros((height, width, 3), dtype=np.uint8)
    aurora[..., 0] = (255 * np.clip(pattern, 0, 1)).astype(np.uint8)  # Red
    aurora[..., 1] = (255 * np.clip(np.sin(pattern * np.pi), 0, 1)).astype(np.uint8)  # Green
    aurora[..., 2] = (255 * np.clip(np.cos(pattern * np.pi / 2), 0, 1)).astype(np.uint8)  # Blue

    return aurora


def process_video():
    # Step 1: Background separation
    cap = cv2.VideoCapture('3762907-uhd_3840_2160_25fps.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Temporary output using XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = 'temp_output.avi'
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    shift = 0  # Initial shift for aurora movement
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Selfie segmentation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(frame_rgb)
            mask = results.segmentation_mask
            condition = mask > 0.5

            # Create black background for foreground
            bg_mask = np.invert(condition).astype(np.uint8)
            fg_mask = condition.astype(np.uint8)

            # Generate aurora pattern with dynamic shift
            aurora_pattern = generate_aurora_pattern(width, height, shift)
            shift += 0.1  # Increment shift for next frame to create movement

            # Apply aurora effect to background only
            aurora_background = cv2.bitwise_and(aurora_pattern, aurora_pattern, mask=bg_mask.astype(np.uint8))
            foreground = cv2.bitwise_and(frame, frame, mask=fg_mask.astype(np.uint8))

            # Combine foreground and aurora background
            final_frame = cv2.add(foreground, aurora_background)

            out.write(final_frame)

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
