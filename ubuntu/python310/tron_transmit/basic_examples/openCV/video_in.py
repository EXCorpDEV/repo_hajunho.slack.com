import sys
import cv2


cap = cv2.VideoCapture('video1.mp4')

if not cap.isOpened():
    print("Video open failed!")
    sys.exit()

print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS:', fps)

delay = round(1000 / fps)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # inversed = ~frame

    cv2.imshow('frame', frame)
    # cv2.imshow('inversed', inversed)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
