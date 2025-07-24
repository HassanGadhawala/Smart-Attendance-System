import cv2
from rembg import remove

video = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("cascades//haarcascade_frontalface_default.xml")

while video.isOpened():

    ret, frame = video.read()
    if ret == True:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(frame_gray, 1.1, 5)

        if len(faces) >= 2 or len(faces) <= 0:
            print(str(len(faces)) + " Invalid")
        else:
            print(str(len(faces)) + " Valid")

        for x, y, w, h in faces:
            # w = w + 50
            # x = x - 25
            # h = h + 50
            # y = y - 25

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi = frame[y:y + h, x:x + w]

            r = remove(roi, bgcolor=(255,0,0,0))
            cv2.imshow("ROI", roi)
            cv2.imshow("Remove", r)

        cv2.imshow("Original", frame)

        k = cv2.waitKey(27)
        if k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
