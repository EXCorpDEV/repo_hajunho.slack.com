import cv2


def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, drawing, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭
        drawing = True
        start_x, start_y = x, y
        img_copy = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing:
            temp_img = img_copy.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow('Select ROI', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        drawing = False
        w = abs(x - start_x)
        h = abs(y - start_y)
        x_coord = min(start_x, x)
        y_coord = min(start_y, y)

        print(f"Bounding Box: x,y,w,h = {x_coord},{y_coord},{w},{h}")

        # Save to bbox.txt
        with open('bbox.txt', 'w') as f:
            f.write(f"{x_coord},{y_coord},{w},{h}")

        print("Coordinates saved to bbox.txt")


# 비디오 파일 열기
video_path = "extest.mp4"  # 여기에 실제 비디오 경로 입력
cap = cv2.VideoCapture(video_path)

# 첫 프레임 읽기
ret, img = cap.read()
if not ret:
    print("Error reading video")
    exit()

# 윈도우 생성 및 마우스 콜백 설정
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', mouse_callback)

# 전역 변수 초기화
drawing = False
start_x, start_y = -1, -1
img_copy = img.copy()

print("Select area with mouse. Press 'q' to quit.")

# 메인 루프
while True:
    cv2.imshow('Select ROI', img_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()