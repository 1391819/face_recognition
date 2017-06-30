import sys
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

cascPath = 'haarcascade_frontalface_alt.xml'

def detect():
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    camera = PiCamera()

    camera.resolution = (640, 480)
    camera.hflip = True
    camera.rotation = 180
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size = (640, 480))

    time.sleep(0.1)

    #Capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        # Grab the raw NumPy array representing the image
        image = frame.array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            img = cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,255), 1)
  
        # Show the frame
        cv2.imshow("Camera", image)
        #cv2.imwrite("capture.pgm", image)
        key = cv2.waitKey(1) & 0xFF

        # Clear the stream so it is ready to receive the next frame
        rawCapture.truncate(0)

        # If the 'q' key was pressed, break from the loop
        if(key == ord('q')):
            break
         
if __name__ == "__main__":
    detect()
