import cv2
import numpy as np

gray = np.zeros((300,300), dtype=np.uint8)

for i in range(300):
    for j in range(300):
        gray[i,j] = int(255 * i / 300) 
        


cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('test.png', a)
