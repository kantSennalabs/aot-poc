from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import imutils
import cv2
import datetime
import os
import time

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gui = Tk(className='Python Examples - Window Size')
# set window size
gui.geometry("1350x800")


# Create a frame
app = Frame(gui, bg="white")
app.grid()
# Create a label in the frame
label = Label( app, text = "Sennalabs")
label.grid(row=0, column=0)
lmain = Label(app)
lmain.grid(row=1, column=0, padx= 95)
input_L = Label(app, text="Enter name").grid(row=2, sticky=W, padx= 630, pady= 10) # label for name
name = Entry(app) # name input box
name.grid(row=3)

# Capture from camera
cap = cv2.VideoCapture("rtsp://admin:Sennalabs_@192.168.0.125/Streaming/Channels/101")
time.sleep(2)
# function for video streaming
def video_stream():
    _, frame = cap.read()
    frame = cv2.resize(frame, (1152 , 648))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    # detect face

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # draw rectangle
    for(x, y, w, h) in faces:
        cv2.rectangle(cv2image, (x, y), (w+x, y+h), (0, 255, 0), 2)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

#snapshot function
def takeSnapshot():
    for i in range(1,11):
        # grab the current timestamp and use it to construct the
        # output path
        _, frame = cap.read()
        outputPath = f'train/{name.get()}'
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime(f'{name.get()}'))
        filename = f'{name.get()}{i}.jpg'
        #p = os.path.sep.join((outputPath, filename))
            # save the file
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        cv2.imwrite(f'./{outputPath}/{filename}', frame.copy())
        print("[INFO] saved {}".format(filename))

btn = Button(app, text="Snapshot!", command=takeSnapshot)
btn.grid(row=4, column=0, pady= 10)
video_stream()
gui.mainloop()

