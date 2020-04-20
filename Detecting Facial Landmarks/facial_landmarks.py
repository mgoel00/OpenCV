from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

modelLoc = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelLoc)

cap =cv2.VideoCapture(0)
ret,image = cap.read()
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):

	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape) #Convert the shape into Numpy array
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #Plot the bounding face
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	for (x, y) in shape: #Mark the landmarks
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
