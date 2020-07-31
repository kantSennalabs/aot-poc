import face_recognition
import cv2
import numpy as np
import datetime

# Load some sample pictures and learn how to recognize them.
img1 = face_recognition.load_image_file("train/pot/pot42.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_encoding = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("train/noom/noom4.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_encoding = face_recognition.face_encodings(img2)[0]

known_faces = [
    # img1_encoding,
    img2_encoding
]

image = cv2.imread('train/noom/noom10.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img3_encoding = face_recognition.face_encodings(image)[0]


begin_time = datetime.datetime.now()

match = face_recognition.compare_faces(known_faces , img3_encoding, tolerance=0.50)
print(match)
print(type(known_faces),type(img3_encoding))
print(datetime.datetime.now() - begin_time)

# now you can compare two encodings
# optionally you can pass threshold, by default it is 0.6
# matches = face_recognition.compare_faces(encodings1, encodings2)