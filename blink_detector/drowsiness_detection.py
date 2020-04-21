from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import winsound

def aspect_ratio(eye):
    return ( dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4]) ) / ( 2 * dist.euclidean(eye[0], eye[3]) )

modelLoc = 'shape_predictor_68_face_landmarks.dat'

EAR_THRESHOLD = 0.28                #It is an experimental value. Might be different for your machine.
COUNTER = 0
print("Loading facial landmark predictor model")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelLoc)
#Extracting the landmarks for both the eyes.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye,rightEye = shape[lStart:lEnd],shape[rStart:rEnd]
            EAR = ( aspect_ratio(leftEye) + aspect_ratio(rightEye) )/2
            if EAR < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= 36:        #To check that the eye was closed for atleast 36 input frames.
                    cv2.putText(frame, "ALERT: BE  AWAKE!!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(2500,duration = 500)
            else:
                COUNTER = 0

        cv2.imshow("Video",frame)
        if cv2.waitKey(1) == 13:
            break

cv2.destroyAllWindows()
cap.release()
