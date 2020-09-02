import cv2
from sklearn import neighbors
import os
import os.path
import sys
import pickle
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from threading import Thread
import datetime
from notification import checkin_teamhero
from rq import Queue
from redis import Redis
import time

font_path = "/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/emojis/Sukhumvit.ttf"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
emotion_model = load_model('/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/model/emotion_recognition.h5')
emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}

emoji = []
for index in range(6):
    emotion = emotions[index]
    emoji.append(cv2.imread('/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/emojis/' + emotion + '.png', -1))

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

# function to return key for any value 
def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
        if val == value: 
            return key 
  
    return "key doesn't exist"

def predict(X_frame, face_list, knn_clf, distance_threshold=0.4):
    predicted_emotion = []
#    if knn_clf is None and model_path is None:
#        raise Exception("Must supply knn classifier either though knn_clf or model_path")
#
#    # Load a trained KNN model (if one was passed in)
#    if knn_clf is None:
#        with open(model_path, 'rb') as f:
#            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)
    print(X_face_locations)

#    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    print("Distance : ",str(min(closest_distances[0])[0]))
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    for top, right, bottom, left in X_face_locations:
        test_image = cv2.resize(cv2.cvtColor(X_frame[top:bottom,left:right], cv2.COLOR_BGR2GRAY), (48, 48))
        test_image = test_image.reshape([-1,48,48,1])
        test_image = np.multiply(test_image, 1.0 / 255.0)
        # Probablities of all classes
        #Finding class probability takes approx 0.05 seconds
        probab = emotion_model.predict(test_image)[0] * 100
        #Finding label from probabilities
        #Class having highest probability considered output label
        label = np.argmax(probab)
        # print('probab_predicted is',probab_predicted)
        predicted_emotion.append(emotions[label])
    # print("predicted_emotion = ",format(predicted_emotion))
    # print("X_face_locations",X_face_locations)

    return [(pred, emotion_list, loc) if rec else ("unknown",emotion_list , loc) for pred, loc, emotion_list, rec  in zip(knn_clf.predict(faces_encodings), X_face_locations, predicted_emotion, are_matches)] 
    


def show_prediction_labels_on_image(img, predictions):
  
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #Upload images of emojis from emojis folder

    for name, emotion, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left-30, top-50), (right+30, bottom+5)), outline=(255, 153, 0))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        text = str(name) + ' is ' + str(emotion)
        text_width, text_height = draw.textsize(text)
        
        font = ImageFont.truetype(font_path, 14)
        draw.rectangle(((left-30, bottom - text_height - 10), (right+30, bottom + 15)), fill=(255, 153, 0), outline=(255, 153, 0))
        
        draw.text((left - 24, bottom - text_height - 4), text,font = font,  fill=(255, 255, 255, 255))
    

    opencvimage = np.array(pil_image)
    for name, emotion, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a label with a name below the face
        text = str(name) + ' is ' + str(emotion)
        text_width, text_height = draw.textsize(text)
        
        emo_top = bottom - 15
        emo_bottom = bottom + text_height + 10
        emo_left = left+text_width + 20
        emo_right = left+text_width+52
        
        frame_cut_size = opencvimage[emo_top:emo_bottom, emo_left:emo_right].shape
#        print(frame_cut_size)
#        print([(emo_top,emo_bottom), (emo_left,emo_right)])
        label = get_key(emotion, emotions)
        emoji_face = emoji[(label)]
        emoji_face = cv2.resize(emoji_face, (frame_cut_size[1], frame_cut_size[0]))
        for c in range(0, 3):
            opencvimage[emo_top-5:emo_bottom - 5, emo_left:emo_right, c] = emoji_face[:, :, c] * \
                (emoji_face[:, :, 3] / 255.0) + opencvimage[emo_top-5:emo_bottom - 5 , emo_left:emo_right, c] * \
                (1.0 - emoji_face[:, :, 3] / 255.0)
        
    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

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
    with open(f"/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/model/trained_knn_model_v{sys.argv[1]}.clf", 'rb') as f:
        knn_clf = pickle.load(f)
            
    cap = CameraVideoStream(src="rtsp://admin:Sennalabs_@192.168.0.63/Streaming/Channels/101").start()
    
    for class_dir in os.listdir("/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/train/"):
        if not os.path.isdir(os.path.join("/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/train/", class_dir)):
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
                try:
                    predictions = predict(img, face_list, knn_clf=knn_clf )
                    if predictions:
                        print("================================================")
                        print(predictions)
                        predicted_name = predictions[0][0]
                        path = f"/home/nvidia-sennalabs/Documents/aot-poc/facial_recognition/cap_img/{predictions[0][0]}"
                        os.makedirs(path, exist_ok=True)
                        img_path = os.path.join(path, f"{predictions[0][0]}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.jpg")
                    
                        cv2.imwrite(img_path , frame)
                        
                        if predicted_name != "unknown":
                            job = q.enqueue(checkin_teamhero, predictions[0][0], img_path, predictions[0][1] ) 
                            cv2.imshow('Found',frame[predictions[0][2][0]*2 - 50:predictions[0][2][2]*2 + 100,predictions[0][2][3]*2 - 100:predictions[0][2][1]*2 + 100])
                except:
                    pass
            try:
                frame = show_prediction_labels_on_image(frame, predictions)
            except:
                pass
            cv2.rectangle(frame, (540, 30),(850, 80), (0,0,0), -1)
            if predicted_name != "unknown":
                cv2.putText(frame, f'{predicted_name} CHECKED IN', (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('camera', frame)
        
            if ord('q') == cv2.waitKey(10):
                cap.stop()
                cv2.destroyAllWindows()
                exit(0)
