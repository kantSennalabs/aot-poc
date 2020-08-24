import cv2
from sklearn import neighbors
import os
import os.path
import sys
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
from threading import Thread
import datetime
from notification import checkin_teamhero
from rq import Queue
from redis import Redis

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

#theading stream
class CameraVideoStream:
    """
    """
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        """
        """
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        """
        """
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        """
        """
        # return the frame most recently read
        return self.frame
    def stop(self):
        """
        """
        # indicate that the thread should be stopped
        self.stopped = True

def predict(X_frame, face_list, knn_clf=None, model_path=None, distance_threshold=0.4):
    print("================================================")
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either though knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    
    X_face_locations = face_recognition.face_locations(X_frame)

#    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    print("Distance : ",str(min(closest_distances[0])[0]))
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)] 
    

def show_prediction_labels_on_image(img, predictions):
  
    pil_image = Image.fromarray(img)
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
    predicted_name = None
    redis_conn = Redis()
    q = Queue(connection=redis_conn)
    process_this_frame = 19
    face_list = []
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 1
    cap = CameraVideoStream(src="rtsp://admin:Sennalabs_@192.168.0.62/Streaming/Channels/101").start()
    
    for class_dir in os.listdir("train/"):
        if not os.path.isdir(os.path.join("train/", class_dir)):
            continue
        face_list.append(class_dir)
        face_list.sort() 

    while True:
        
        frame = cap.read()
        if cap.grabbed:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0,0), fx= 0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 20 == 0:
                predictions = predict(img, face_list, model_path= f"model/trained_knn_model_v{sys.argv[1]}.clf")
                print(predictions)
                if predictions:
                    predicted_name = predictions[0][0]
                    path = f"cap_img/{predictions[0][0]}"
                    os.makedirs(path, exist_ok=True)
                    img_path = os.path.join(path, f"{predictions[0][0]}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.jpg")
                    try:
                        cv2.imwrite(img_path , frame)
                    except:
                        pass
                    job = q.enqueue(checkin_teamhero, predictions[0][0], img_path)
                    try:
                        cv2.imshow('Found',frame[predictions[0][1][0]*2 - 50:predictions[0][1][2]*2 + 100,predictions[0][1][3]*2 - 100:predictions[0][1][1]*2 + 100])
                    except:
                        pass
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.rectangle(frame, (540, 30),(850, 80), (0,0,0), -1)
            cv2.putText(frame, f'{predicted_name} CHECKED IN', (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA) if predicted_name != "unknown" else None
            cv2.imshow('camera', frame)
        
            if ord('q') == cv2.waitKey(10):
                cap.stop()
                cv2.destroyAllWindows()
                exit(0)
