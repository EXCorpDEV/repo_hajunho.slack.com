from ultralytics import YOLO

def run_yolo_inference():
    # 사전 학습된 YOLOv8 모델을 로드
    model = YOLO('yolov8n.pt')  # n, s, m, l, x 모델 등 원하는 것 선택

    # 로컬 이미지나 URL 이미지에 대해 추론
    results = model('https://ultralytics.com/images/bus.jpg', show=False)

    # 결과 분석
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])       # 클래스 인덱스
            conf = float(box.conf[0])     # confidence
            print(f"Detected class: {model.names[cls_id]} ({conf:.2f})")

if __name__ == "__main__":
    run_yolo_inference()
