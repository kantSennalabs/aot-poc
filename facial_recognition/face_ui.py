from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import imutils
import cv2
import datetime
import os

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
lmain.grid(row=1, column=0, padx= 35)
input_L = Label(app, text="Enter name").grid(row=2, sticky=W)

# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    _, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

#snapshot function
def takeSnapshot():
		# grab the current timestamp and use it to construct the
		# output path
        _, frame = cap.read()
        outputPath = '/'
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((outputPath, filename))
        # save the file
        cv2.imwrite(f'./{filename}', frame.copy())
        print("[INFO] saved {}".format(filename))

btn = Button(app, text="Snapshot!", command=takeSnapshot)
btn.grid(row=2, column=0)
video_stream()
gui.mainloop()

