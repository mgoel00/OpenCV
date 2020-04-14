import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
params = {"prototxt":"prototxt.txt", "model":"face_detect_model.caffemodel", "confidence":0.5}
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(params["prototxt"], params["model"])

cap = cv2.VideoCapture(0)
ret,image = cap.read()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > params["confidence"]:
		# compute the (x, y)-coordinates of the bounding box
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
cap.release()
