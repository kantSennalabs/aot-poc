import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

def predict(X_frame, face_list, knn_clf=None, model_path=None, distance_threshold=0.45):
    print("================================================")
    acc_list = []
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either though knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    # closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # print("Distance : ",str(min(closest_distances[0])[0]))
    # are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # props = knn_clf.predict_proba(faces_encodings)
    # if are_matches:
    #     for item in props:
    #         item = item.tolist()
    #         for idx in range(len(item)):
    #             if item[idx] != 0:
    #                 # print(face_list[idx]+" : "+str(item[idx]))
    #                 acc_list.append((face_list[idx],item[idx]))
    #     # print(acc_list)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc)  for pred, loc in zip(knn_clf.predict(faces_encodings), X_face_locations)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.
    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":

    process_this_frame = 9
    face_list = []
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 1
    # cap = cv2.VideoCapture("rtsp://admin:HuaWei123@192.168.1.75:554/LiveMedia/ch1/Media2")
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("rtsp://admin:senna1234@192.168.1.33:554")
    
    for class_dir in os.listdir("train/"):
        if not os.path.isdir(os.path.join("train/", class_dir)):
            continue
        face_list.append(class_dir)
        face_list.sort() 

    while 1 > 0:
        ret, frame = cap.read()
        # frame = cv2.flip(frame, -1)
        # print(ret,frame)
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 10 == 0:
                predictions = predict(img, face_list, model_path="trained_knn_model.clf")
                print(predictions)
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)