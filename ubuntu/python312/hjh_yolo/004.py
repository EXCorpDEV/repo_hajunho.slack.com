import os
import requests
import cv2
from ultralytics import YOLO

# 다른 샘플 MP4 URL
VIDEO_URL = "https://www.pexels.com/ko-kr/download/video/3873059/?fps=25.0&h=1080&w=1920"
VIDEO_FILE = "3873059-hd_1920_1080_25fps.mp4"


def download_sample_video(url, filename):
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Skipping download.")
        return

    print(f"Downloading sample video from {url}...")
    response = requests.get(url, stream=True)

    # HTTP 상태 코드 체크 (200이 아닐 경우 예외)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status code {response.status_code}.")

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.\n")


def detect_video(input_path='sample_video.mp4', output_path='result_video.mp4', show=False):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video source: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("=== Starting YOLO detection ===")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        if show:
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    out.release()
#    cv2.destroyAllWindows()
    if show:
        cv2.destroyAllWindows()
    print(f"Detection completed. Results saved to '{output_path}'")


def main():
    #download_sample_video(VIDEO_URL, VIDEO_FILE)
    detect_video(input_path='2024futureshow.mp4', output_path='24futershow.mp4', show=False)


if __name__ == "__main__":
    main()
