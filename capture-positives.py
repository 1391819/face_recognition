# Importing needed classes and libraries
import glob
import os
from os import listdir
from os.path import isfile, join
import sys
import select
import cv2
import picam
import sqlite3

# Declaring the xml file used for the CascadeClassifier object
# We are going to use the haar one because it is more accurate
# Even though it is slower compared to the lbp one
haar_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#lbp_faces = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

def detect_single(image):

	"""Return bounds (x, y, width, height) of detected face in grayscale image.
	   If no face or more than one face are detected, None is returned.
	"""
	# Creating detectMultiScale object which helps us retrive faces in images
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
	return cv2.resize(image,(92, 112), interpolation=cv2.INTER_LANCZOS4)

if __name__ == '__main__':

        # It's always preferable to perform database operations inside the try-catch clause
        # in order to handle errors and transactions using the commit-rollback keywords
        try:
                
                # List containing all images
                all_files = []

                # Database connection
                db = sqlite3.connect('dataset.db')

                # Declaring cursor used in sqlite to perform CRUD operations on the database
                cursor = db.cursor()

                # Creating the table in the database if it doesn't exist already
                cursor.execute(''' CREATE TABLE IF NOT EXISTS users(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name VARCHAR(60) NOT NULL,
                   filepath TEXT NOT NULL
                   )

                ''')

                # Declaring OpenCVCapture object from the picam class
                camera = picam.OpenCVCapture()

                # Create the directory for positive training images if it doesn't exist.
                if not os.path.exists('./training/dataset'):
                        os.makedirs('./training/dataset')

                # Entering username from keyboard
                name = raw_input('Enter your username: ')
                
                # Find the largest ID of existing positive images.
                # Start new images after this ID value.
                files = sorted(glob.glob(os.path.join('./training/dataset', 
                        '[0-9][0-9][0-9]' + name + '[0-9][0-9][0-9].pgm')))

                # Grabbing XXX numbers to put before and after the name in the filename ( doing this just for having an ordered directory with all the people )
                count = 0
                label = 0
                
                if len(files) > 0:
                        
                        # Grab the count from the last filename.
                        count = int(files[-1][-7:-4])+1
                        label = int(files[-1][19:22])
                else:
                        try:
                                
                                # Grabbing all files from the given directory
                                temp_all_files = [f for f in listdir('./training/dataset') if isfile(join('./training/dataset', f))]

                                # 'Cutting' the filename and taking the first 3 numbers 
                                for file in temp_all_files:
                                        all_files.append(file[:3])

                                # Ordering all numbers in the list
                                all_files.sort()

                                # Taking the last label from the list
                                last_label = all_files[-1]

                                # Adding one to the number to obtain the new label        
                                label = int(last_label) + 1
                                
                        except IndexError:
                                # If there aren't any photos in the directory so not even a label is found, last_label is set to 0
                                label = 0
                        

                # Entering amount of pictures we want to take
                c = raw_input('Enter number of pictures you want to take: ')

                # Count variables used for the while loop
                intC = int(c)
                k = 0
                
                print 'Capturing training images...'
   
                while k < intC:
                        
                        print 'Capturing image...'
                        # Getting image from the camera and the picam class
                        image = camera.read()
                                
                        # Convert image to grayscale.
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                        # Get coordinates of single face in captured image.
                        result = detect_single(image)

                        # If no face jas been found
                        if result is None:
                                print 'Could not detect single face!'
                                continue
                        
                        # Getting faces coordinates
                        x, y, w, h = result
                                
                        # Crop image as close as possible to desired face aspect ratio.
                        crop_image = resize(crop(image, x, y, w, h))
                                
                        # Save image to file.
                        filename = os.path.join('./training/dataset', '{0:03}{1}{2:03}.pgm'.format(label, name, count))
                        cv2.imwrite(filename, crop_image)

                        # Inserting into the table users the new user
                        cursor.execute('INSERT INTO users(name, filepath) VALUES(?, ?)', (name, filename))

                        # Committing changes to the database
                        db.commit()
                        
                        print 'Found face and wrote training image', filename

                        # Incrementing variable used to name user's filename (e.g 000Nacu000 -> 000Nacu001 -> 000Nacu002 and so on)
                        count += 1
                        
                        # Count variables used for the while loop      
                        k += 1

        # Handling database exceptions
        except sqlite3.Error as e:
                raise e
                # Rollback if there were errors
                db.rollback()
                
        # Finally clause important because it closes the database connection everytime
        finally:
                db.close()

