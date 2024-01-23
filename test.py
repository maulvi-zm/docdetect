import docdetect
import cv2

im = cv2.imread("2.jpg")
rects = docdetect.process(im)
im = docdetect.draw(rects, im)

cv2.imshow("output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()