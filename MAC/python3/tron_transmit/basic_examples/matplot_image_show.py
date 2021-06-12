import cv2 #BGR color sequence.
from matplotlib import pyplot as plt #RGB color sequence

image = 'images/a.jpg'

img = cv2.imread(image)
plt.axis('off')

#Color Space Conversion
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.show()
