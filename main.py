import sys
from cv2 import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCasc = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    if frame is not None:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCasc.detectMultiScale(
            gray,
            # scaleFactor = 1.0,
            # minNeighbors = 5,
            # minSize = (30,30),
            # flags = CASCADE_SCALE_IMAGE
        )

        font = cv2.FONT_ITALIC

        cv2.putText(frame,
                    'Press q to exit',
                    (0, 25),
                    font, 1,
                    (0, 255, 2),
                    2,
                    cv2.LINE_AA)

        cv2.putText(frame,
                    'Face detected:'+str(len(faces)),
                    (0, 75),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:

        print("Empty frame")
        exit(1)

video_capture.release()
cv2.destroyAllWindows()
