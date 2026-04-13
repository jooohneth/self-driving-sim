import numpy as np
import pandas as pd
import glob
import cv2
import mtcnn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')



face_detector = mtcnn.MTCNN()


# DATA
def face_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try: 
        x, y, w, h = face_detector.detect_faces(img)[0]['box']
        return img[x:x+w, y:y+h]
    except:
        pass


data_list = []
label_list = []

for i, address in enumerate(glob.glob("Session18/smile_dataset/smile_dataset\\*\\*")):
    print(address)
    image = cv2.imread(address)
    # cv2.imshow('frame', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = face_detection(image)
    image = cv2.resize(image, (32,32))
    image = image/255
    image = image.flatten()

    data_list.append(image)

    label_list.append(address.split('\\')[1])

    if i%200==0:
        print(f'[INFO] {i}/3600 images processed!')


X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

# EVALUATE
predictions = model.predict(X_test)
print(f'accuracy={accuracy_score(y_test, predictions)*100}')

joblib.dump(model, 'Session18/smile_detection.z')