import cv2
from rembg import remove

try:
    img = cv2.imread("img_3.png")
    img = cv2.resize(img, (512, 512))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier("cascades//haarcascade_frontalface_default.xml")

    faces = cascade.detectMultiScale(img_gray,1.1,5)
    for x, y, w, h in faces:
        w = w + 50
        x = x - 25
        h = h + 50
        y = y - 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x:x + w]

        r = remove(roi, alpha_matting=True, bgcolor=(255, 0, 55, 1))

        smooth = cv2.bilateralFilter(cv2.cvtColor(r, cv2.COLOR_BGRA2BGR), 7, 80, 80)
        br = out = cv2.addWeighted(smooth, 1.2, smooth, 0, 1.2)
        cv2.imshow("ROI", roi)
        cv2.imshow("Remove", r)
        cv2.imshow("SMOOTH", smooth)
        cv2.imshow("Brightness", br)

    cv2.imshow("Original", img)

except:
    print("Something Went Wrong.... Please Try again........")
cv2.waitKey(0)
cv2.destroyAllWindows()