import io
import time
import cv2
import numpy as np
import picamera

class OpenCVCapture(object):
        
	def read(self):
                
		"""Read a single frame from the camera and return the data as an OpenCV
		image (which is a numpy array).
		"""
		
		# Capture a frame from the camera.
		data = io.BytesIO()
		
		with picamera.PiCamera() as camera:
                        camera.brightness = 60
                        camera.rotation = 180
			camera.resolution = (640, 480)
			camera.capture(data, format='jpeg')
			
		data = np.fromstring(data.getvalue(), dtype=np.uint8)
		
		# Decode the image data and return an OpenCV image.
		image = cv2.imdecode(data, 1)
		
		# Save captured image for debugging.
		cv2.imwrite('capture.pgm', image)
		
		# Return the captured image data.
		return image

		
