import glob
import os
from os import listdir
from os.path import isfile, join
import sys
import select
import cv2
import picam

def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False

haar_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#haar_faces = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')


def detect_single(image):

	"""Return bounds (x, y, width, height) of detected face in grayscale image.
	   If no face or more than one face are detected, None is returned.
	"""
	faces = haar_faces.detectMultiScale(image, 
				scaleFactor= 1.2, 
				minNeighbors= 2, 
				minSize= (30, 30), 
				flags=cv2.CASCADE_SCALE_IMAGE)
	if len(faces) != 1:
		return None
		
	return faces[0]

def crop(image, x, y, w, h):

	"""Crop box defined by x, y (upper left corner) and w, h (width and height)
	to an image with the same aspect ratio as the face training data. 
	"""
	crop_height = int((112 / float(92)) * w)
	midy = y + h/2
	y1 = max(0, midy-crop_height/2)
	y2 = min(image.shape[0]-1, midy+crop_height/2)
	return image[y1:y2, x:x+w]

def resize(image):

	"""Resize a face image to the proper size for training and detection.
	"""
	return cv2.resize(image, 
					  (92, 112), 
					  interpolation=cv2.INTER_LANCZOS4)


if __name__ == '__main__':

        all_files = []
        
	camera = picam.OpenCVCapture()

	# Create the directory for positive training images if it doesn't exist.
	if not os.path.exists('./training/dataset'):
		os.makedirs('./training/dataset')

        name = raw_input('Enter your username: ')
	# Find the largest ID of existing positive images.
	# Start new images after this ID value.
	files = sorted(glob.glob(os.path.join('./training/dataset', 
		'[0-9][0-9][0-9]' + name + '[0-9][0-9][0-9].pgm')))

        #print files

        
	count = 0
        label = 0
	if len(files) > 0:
		# Grab the count from the last filename.
		count = int(files[-1][-7:-4])+1
		#print count
		label = int(files[-1][19:22])
                #print label
        else:
                try:
                        
                        temp_all_files = [f for f in listdir('./training/dataset') if isfile(join('./training/dataset', f))]

                        for file in temp_all_files:
                                #print file[:3]
                                all_files.append(file[:3])
                        
                        all_files.sort()
                        last_label = all_files[-1]
                        
                        #print last_label                
                        label = int(last_label) + 1
                except IndexError:
                        last_label = 'null'
                

        c = raw_input('Enter number of pictures you want to take: ')
	intC = int(c)	
	print 'Capturing positive training images...'

        k = 0
  
        while k < intC:
		print 'Capturing image...'
		image = camera.read()
			
		# Convert image to grayscale.
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		# Get coordinates of single face in captured image.
		result = detect_single(image)
			
		if result is None:
                        print 'Could not detect single face!'
			continue
		x, y, w, h = result
			
		# Crop image as close as possible to desired face aspect ratio.
		crop_image = crop(image, x, y, w, h)
			
		# Save image to file.
		filename = os.path.join('./training/dataset', '{0:03}{1}{2:03}.pgm'.format(label, name, count))
		cv2.imwrite(filename, crop_image)
			
		print 'Found face and wrote training image', filename
			
		count += 1
		k += 1
