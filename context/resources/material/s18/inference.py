import joblib
import cv2
from mtcnn import MTCNN

face_detector = MTCNN()

def face_detection(img):
    try: 
        x, y, w, h = face_detector.detect_faces(img)[0]['box']
        return img[x:x+w, y:y+h]
    except:
        pass

model = joblib.load("Session18/model.z")
image = cv2.imread("image")
image = face_detection(image)
image = cv2.resize(image, (32,32))
image = image/255
image = image.flatten()

print(model.predict([image]))
