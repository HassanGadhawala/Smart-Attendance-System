import cv2
import os
from PIL import Image
import numpy as np

model = cv2.face.LBPHFaceRecognizer_create()

def getFacesIDS():
    paths = []
    faces = []
    IDs = []
    folder = "dataset"

    for i in os.listdir(folder):
        paths.append(os.path.join(folder, i))

    for path in paths:
        read_image = Image.open(path).convert('L').resize((200,200),Image.LANCZOS)
        face_numpy = np.asarray(read_image, 'uint8')
        ID = int(os.path.split(path)[1].split('_')[0])
        faces.append(face_numpy)
        IDs.append(ID)

        cv2.imshow("Numpy Image", face_numpy)
        cv2.waitKey(27)

    return faces, IDs


Faces, IDs = getFacesIDS()

try:
    model.train(Faces, np.array(IDs))
    model.save("Trained_Model.yml")
except Exception as ex:
    print("An Exception occurred!!!",ex)

cv2.destroyAllWindows()