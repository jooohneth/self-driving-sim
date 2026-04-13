import cv2

model = cv2.CascadeClassifier("Session9\haarcascades\haarcascade_frontalface_default.xml")

img = cv2.imread("Session9\S11_face_college.jpg")

predictions = model.detectMultiScale(img, 1.1, 51)

for x, y, w, h in predictions:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()