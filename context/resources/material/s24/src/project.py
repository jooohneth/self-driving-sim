from preprocessing import data_preprocessing
from model import train
from evaluation import result_vis

EPOCHS = 10
# BATCH_SIZE = 


X_train, X_test, y_train, y_test = data_preprocessing()
H = train(X_train, X_test, y_train, y_test, EPOCHS)
result_vis(H, EPOCHS)
