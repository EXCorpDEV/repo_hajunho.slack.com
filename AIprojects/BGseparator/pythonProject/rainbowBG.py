import cv2
import mediapipe as mp
import numpy as np
import subprocess


def create_rainbow_overlay(width, height):
    """무지개 그라데이션 생성"""
    rainbow = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (148, 0, 211),  # 보라색
        (75, 0, 130),  # 남색
        (0, 0, 255),  # 파랑
        (0, 255, 0),  # 초록
        (255, 255, 0),  # 노랑
        (255, 127, 0),  # 주황
        (255, 0, 0)  # 빨강
    ]

    stripe_height = height // len(colors)
    for i, color in enumerate(colors):
        cv2.rectangle(
            rainbow,
            (0, i * stripe_height),
            (width, (i + 1) * stripe_height),
            color,
            -1
        )
    return rainbow


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

    rainbow_overlay = create_rainbow_overlay(width, height)

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
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

            # Apply rainbow to background only
            rainbow_background = cv2.bitwise_and(rainbow_overlay, rainbow_overlay, mask=bg_mask.astype(np.uint8))
            foreground = cv2.bitwise_and(frame, frame, mask=fg_mask.astype(np.uint8))

            # Combine foreground and rainbow background
            final_frame = cv2.add(foreground, rainbow_background)

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
