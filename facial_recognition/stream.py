import numpy as np
import cv2
import sys

port = [62,63]

cap = cv2.VideoCapture(f"rtsp://admin:Sennalabs_@192.168.0.{port[1]}/Streaming/Channels/101")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

