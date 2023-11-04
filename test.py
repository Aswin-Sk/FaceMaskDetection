from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import face
model = load_model('mask.model')
confidence = 0.5

def predict_mask(image_file):
    image = cv2.imread(image_file)
    number,cut_face=face.detect_face(image)
    if number==0:
        return "No face found"
    elif number==-1:
        return "Multiple faces found"
    cut_face = cv2.resize(cut_face, (224, 224))
    cut_face = cv2.cvtColor(cut_face, cv2.COLOR_BGR2RGB)
    cut_face = np.expand_dims(cut_face, axis=0)
    cut_face = preprocess_input(cut_face)
    prediction = model.predict(cut_face)
    if prediction[0][1] < confidence:
        return "Mask detected"
    else:
        return "Mask not detected"
image_path = 'test_image.jpg'
predict_mask(image_path)
