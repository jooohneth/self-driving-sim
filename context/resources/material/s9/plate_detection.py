import cv2


model = cv2.CascadeClassifier("Session9\haarcascades\haarcascade_russian_plate_number.xml")
cap = cv2.VideoCapture("Session4\S10_cars_video (1).mp4")

while True:

    flag, frame = cap.read()

    if not flag:
        break

    frame = cv2.resize(frame, (800, 600))
    predictions = model.detectMultiScale(frame, 1.1, 5)

    for x, y, w, h in predictions:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()