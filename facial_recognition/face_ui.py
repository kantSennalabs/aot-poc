from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import imutils
import cv2


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

video_stream()
gui.mainloop()

