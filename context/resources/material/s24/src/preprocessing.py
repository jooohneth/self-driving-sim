import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def data_preprocessing():
    print('[INFO] DATA IS BEING PROCESSED...')
    label_list = []
    data_list = []
    for i, address in enumerate(glob.glob("Session24/src/data/dataset\\*\\*")):
        image = cv2.imread(address)
        image = cv2.resize(image, (244, 244))
        image = image/255

        data_list.append(image)
        label_list.append(address.split("\\")[1])

    X = np.array(data_list)
    y = np.array(label_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    return X_train, X_test, y_train, y_test




        
