import sys
import cv2

cap = cv2.VideoCapture(0) #camera number or video file

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# Camera Frame Size
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    inversed = ~frame

    # cv2.imshow('frame', frame) #normal
    cv2.imshow('inversed', inversed)

    if cv2.waitKey(10) == 27: #ESC
        break

cap.release()
cv2.destroyAllWindows()
