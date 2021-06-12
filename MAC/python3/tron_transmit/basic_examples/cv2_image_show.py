import cv2

img = cv2.imread("cat.bmp")
img2 = cv2.imread("images/a.jpg")
img3 = cv2.imread('images/b.jpg')

cv2.namedWindow("img")
cv2.namedWindow('img2')
cv2.namedWindow("img3")

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey()
print("wait done")
cv2.destroyWindow('img')
cv2.waitKey()
