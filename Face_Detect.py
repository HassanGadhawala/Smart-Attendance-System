import cv2
import os
import dlib
import numpy as np

#==================Detect Faces on Webcam===========================
if os.path.exists("dataset") == False:
    os.makedirs("dataset")

user_id = str(input("Enter User ID : "))
count = 0
folder = "dataset"
video = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()

while video.isOpened():

    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame)

    for face in faces:

        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        roi = frame_gray[y: y1, x: x1]

        cv2.rectangle(frame, (x, y), (x1, y1), (255, 255, 0), 2)

        # cv2.rectangle(frame, (x + 20, y + 20), (x + w - 20, y + h - 20), (255, 0, 0), 2)
        # roi = frame_gray[y + 20: y + h - 20, x + 20: x + w - 20]

        count = count + 1

        roi = cv2.resize(roi,(200,200))

        cv2.imwrite(f"{folder}//{user_id}_{count}.png",roi)

        cv2.putText(frame,str(count),(25,25),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)

        cv2.imshow("ROI ", roi)



    cv2.imshow("Video",frame)

    k = cv2.waitKey(27)
    if k == ord('q'):
        break
    elif count == 70:
        break

video.release()
cv2.destroyAllWindows()
