import cv2
haar_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_face = frame[y:y + h, x:x + w]
        return 1,cropped_face
    elif len(faces)==0:
        return 0,None
    else:
        return -1,None

