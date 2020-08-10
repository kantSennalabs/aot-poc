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
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        i = 0
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image,known_face_locations=face_bounding_boxes)[0])
                # print(X)
                y.append(class_dir)
                i += 1
        # print(f"{class_dir} : {i}")
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    # print(X[2].shape)
    # Create and train the KNN classifier
    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(X, y)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    mlp_clf = MLPClassifier(random_state=1, max_iter=300)

    mlp_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(mlp_clf, f)

    return mlp_clf


def test(test_dir, model):
    
    X_test = []
    y_test = []

    # Loop through each person in the training set
    for class_dir in os.listdir(test_dir):
        i =0
        if not os.path.isdir(os.path.join(test_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(test_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                pass
                # If there are no people (or too many people) in a training image, skip the image.
                # if verbose:
                #     print("Image {} not suitable for te: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X_test.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y_test.append(class_dir)
                i += 1
        print(f"{class_dir} : {i}")
    # Determine how many neighbors to use for weighting in the KNN classifier
    # for i, item in enumerate(X_test):
    #     closest_distances = model.kneighbors(X_test, n_neighbors=1)
    #     print(str(i)," : ",str(closest_distances))

    # print(X[2].shape)
    # Create and train the KNN classifier
    y_pred = model.predict(X_test)
    print('========================')
    for idx, item in enumerate(y_pred):
        print(y_test[idx],y_pred[idx])
    acc = accuracy_score(y_test, y_pred)

    return acc




if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("train/", model_save_path="trained_knn_model.clf", verbose=True)
    print("Training complete!")
    print("Testing ....")
    acc = test("test/",classifier)
    print(acc)

