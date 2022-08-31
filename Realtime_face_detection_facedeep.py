import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
#chek if the webcam is opend correctly

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")
#read one image from video
while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame, actions= ['emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    #Draw a rectangle around the faces 
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    #insert text on video
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (20, 20), font, 1, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow('original video', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()