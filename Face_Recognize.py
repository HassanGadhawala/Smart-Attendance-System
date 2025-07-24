import cv2
import numpy as np
import dlib

model = cv2.face.LBPHFaceRecognizer_create()

model.read("Trained_Model.yml")

video = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()

while video.isOpened() is not None:
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        roi = frame_gray[y: y1, x: x1]

        cv2.rectangle(frame, (x, y), (x1, y1), (255, 255, 0), 2)

        label, confidence = model.predict(roi)
        # conf = int((100*(1-confidence/180)))
        print(label,confidence)

        # if conf > 77:
        #     cv2.putText(frame,str(label),(x,y - 10),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255))
        # else:
        #     cv2.putText(frame,"Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
        cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))

    cv2.imshow("Video",frame)

    k = cv2.waitKey(27)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
