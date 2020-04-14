import numpy as np
import imutils
import cv2

params = {"prototxt":"prototxt.txt", "model":"face_detect_model.caffemodel", "confidence":0.5}
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(params["prototxt"], params["model"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage( cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0) )

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	print(detections.shape)
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < params["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for object

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) == 13:
		break

cv2.destroyAllWindows()
vs.release()
