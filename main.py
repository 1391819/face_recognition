#!/usr/bin/python
# -*- coding: utf8 -*-

# Importing needed classes and libraries
import io
import numpy as np 
import RPi.GPIO as GPIO 
import time
import cv2 
import fnmatch
import sys
import subprocess #####################
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import os
from gtts import gTTS #######################
import datetime
import calendar
import sqlite3

# Declaring the xml file used for the CascadeClassifier object
# We are going to use the haar one because it is more accurate
# Even though it is slower compared to the lbp one
haar_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#lbp_faces = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

#-----------------------------------------------------------------------------------------------

# We are going to unify picamera and cv2.VideoCapture into a single class with openCV
# This new class, used altogether with threading, could increase our FPS processing rate
class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32, rotation=180, hflip=False, vflip=False):
        
        # Initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.rotation = rotation
        self.camera.framerate = framerate
        self.camera.hflip = hflip
        self.camera.vflip = vflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)

        # Initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        
        # Start the thread to read frames from the video stream
        t = Thread(target=self.update, args=()) 
        t.daemon = True
        t.start()
        return self

    def update(self):
        
        # Keep looping infinitely until the thread is stopped
        for f in self.stream:
            
            # Grab the frame from the stream and clear the stream in preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # If the thread indicator variable is set, stop the thread and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        
#-----------------------------------------------------------------------------------------------
        
# Function used to retrieve the current_time of the Raspberry Pi
def get_current_time():
    pm = False
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute

    # Convert to 12 hour time
    if hour == 0:
        hour = 12
    elif hour > 11:
        pm = True
        hour -= 12

    # If single digit minute, add '0' to the beginning
    if minute < 10:
        minute = "0" + str(minute)
    
    return str(hour) + " " + str(minute) + " " + ("PM" if pm else "AM")

#-----------------------------------------------------------------------------------------------

# Function used to create an mp3 file containing a certain text    
def download(message, outfile='message.mp3'):
    message = gTTS(text=message, lang='it') 
    message.save(outfile)

#-----------------------------------------------------------------------------------------------

# Class which handles the button signal
class button(object):
           
	def __init__(self):
            
		# Initialize button.
		GPIO.setmode(GPIO.BCM)  
		GPIO.setup(25, GPIO.IN)
		
		# Set initial button state.
		self.button_state = GPIO.input(25)

	def is_button_up(self):
            
		#Return True when the button has transitioned from down to up (i.e. the button was pressed).
		old_state = self.button_state
		self.button_state = GPIO.input(25)
		
		# Check if transition from down to up
		if old_state == False and self.button_state == True:
                    
			# Wait 20 milliseconds and measure again to debounce switch.
			time.sleep(20.0/1000.0)
			self.button_state = GPIO.input(25)
			if self.button_state == True:
				return True
		return False
	    
        # Cleaning everything
	def end(self):
                GPIO.cleanup()
	    
#-----------------------------------------------------------------------------------------------
        
def walk_files(directory, match='*'):

	# Generator function to iterate through all files in a directory recursively which match the given filename match parameter.
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, match):
			yield os.path.join(root, filename)
			
#-----------------------------------------------------------------------------------------------
			
def prepare_image(filename):

	# Read an image as grayscale and resize it to the appropriate size for training the face recognition model.
	return resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
	
#-----------------------------------------------------------------------------------------------

# Small function used to delete duplicates in a list			
def delete_duplicates(seq, idfun=None):
        # Order preserving
        if idfun is None:
                def idfun(x):
                        return x
                seen = {}
                result = []
                for item in seq:
                        marker = idfun(item)
                        if marker in seen:
                                continue
                        seen[marker] = 1
                        result.append(item)
                return result
            
#------------------------------------------------------------------------------------------------------------------
def update_database(database, cursor):

    # Creating the table in the database if it doesn't exist already
    cursor.execute(''' CREATE TABLE IF NOT EXISTS users(

                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(60) NOT NULL,
                        filepath TEXT NOT NULL
                   )
                ''')

    # Deleting all previous table content
    cursor.execute('DELETE FROM users')
                
    # Committing changes to the database
    database.commit()

    # Read all training images
    for filename in walk_files('./training/dataset', '*.pgm'):

        temp_file = os.path.basename(filename)
        username = temp_file[3:-7]
        filename = './training/dataset/' + temp_file

        # Inserting into the table the user found from the filename
        cursor.execute('INSERT INTO users(name, filepath) VALUES(?, ?)', (username, filename))

        # Committing changes to the database
        database.commit()    

#------------------------------------------------------------------------------------------------------------------
    
def crop(image, x, y, w, h):

	"""Crop box defined by x, y (upper left corner) and w, h (width and height)
	to an image with the same aspect ratio as the face training data. 
	"""
	crop_height = int((112 / float(92)) * w)
	midy = y + h/2
	y1 = max(0, midy-crop_height/2)
	y2 = min(image.shape[0]-1, midy+crop_height/2) 
	return image[y1:y2, x:x+w]
    
#------------------------------------------------------------------------------------------------------------------
    
def resize(image): 

	"""Resize a face image to the proper size for training and detection. 
	"""
	# Lanczos resampling is typically used to increase the sampling rate of a digital signal, or to shift it
	# by a fraction of the sampling interval. It is often used also for multivariate interpolation, for example to
	# resize or rotate a digital image.
	return cv2.resize(image, 
					  (92, 112), 
					  interpolation=cv2.INTER_LANCZOS4)
    
#------------------------------------------------------------------------------------------------------------------
    
# Function used to retrieve tha amount of fps
def show_FPS(start_time,frame_count):
    
    if frame_count >= 100:
        duration = float(time.time() - start_time)
        FPS = float(frame_count / duration)
        print("Processing at %.2f fps last %i frames" %( FPS, frame_count))
        frame_count = 0
        start_time = time.time()
        
    else:
        frame_count += 1
        
    return start_time, frame_count

#-----------------------------------------------------------------------------------------------

# Function used to train the algorithm
def train(database, cursor, model):

    print "Reading training images..."

    # Lists used to contain data used to train the algorithm
    faces = []
    labels = []
    people_count = 0

    # Retrieving all ids and filepaths from the users table
    cursor.execute('SELECT id, filepath FROM users')

    # Getting all found rows in the database using the query above
    all_rows = cursor.fetchall()

    # For each row in all rows found
    for row in all_rows:
        # Getting single label and filepath
        label = row[0]
        file = row[1]

        # Append found data to the previously created lists
        # Using prepare_image(file) in order to return the resized image (92, 112) 
        faces.append(prepare_image(file))
        labels.append(label)

        # Incrementing amount of images found
        people_count += 1

    print 'Read', people_count, 'people.'

    # Training model
    print 'Training model...'
    #model = cv2.face.createEigenFaceRecognizer() 
    model.train(np.asarray(faces), np.asarray(labels)) 

    # Saving model results
    model.save('training.xml') #####################
    print 'Training data saved to training.xml!'

    # Save mean and eignface images which summarize the face recognition model.
    # https://github.com/MasteringOpenCV/code/blob/master/Chapter8_FaceRecognition/recognition.cpp
    # class face_BasicFaceRecognizer http://docs.opencv.org/master/dc/dd7/classcv_1_1face_1_1BasicFaceRecognizer.html
    # http://www.bytefish.de/blog/eigenvalues_in_opencv/
    # https://codeyarns.com/2015/06/01/eigenvectors-and-eigenvalues-in-opencv/
    # https://nibblesandbytes.wordpress.com/2011/04/26/eigenvalues-of-an-image-opencv/
    # http://math.mit.edu/~gs/linearalgebra/ila0601.pdf
    # Show the average face (statistical average for each pixel in the collected images).
	
    mean = model.getMean() 
    eigenvectors = model.getEigenVectors() 

    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(faces[0].shape)
    cv2.imwrite('mean.png', mean_resized)

    # Turn the first and best (at most) 20 eigenvectors into grayscale images. Eigenvectors are stored by column.
    for i in xrange(min(len(faces), 20)):
        eigenvector_i = eigenvectors[:, i].reshape(faces[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        cv2.imwrite('eigenvectors/eigenface_%d.png' % i, eigenvector_i_norm)


    eigenvalues = model.getEigenValues()
    #print eigenvalues
    
    projections = model.getProjections()
    #print projections
	
#-----------------------------------------------------------------------------------------------

def normalize(X, low, high, dtype=None): 
	"""Normalizes a given array in X to a value between low and high.
	Adapted from python OpenCV face recognition example at:
	  https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
	"""
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

#------------------------------------------------------------------------------------------------------------------

# Function used to output a greeting message when a known face is found	
def user_known_audio(username):
    
    greeting = "Buona giornata " + username + "."
    time = "Sono le " + get_current_time() + ". "

    download(greeting + time, outfile="audio_sample/greeting.mp3")
    #download(greeting, outfile="audio_sample/greeting.mp3")
    subprocess.call("avplay -nodisp -autoexit audio_sample/greeting.mp3", shell=True) ######################
    os.remove("audio_sample/greeting.mp3")

#------------------------------------------------------------------------------------------------------------------
  
# Main function used for face recognition
def face_recognition(database, cursor, model):

    # Loading training data into the model using the .xml file previously created 
    print 'Loading training data...'
    model.load('training.xml') 
    print 'Training data loaded!'

    # Initializing camera using threads
    print("Initializing Camera ....")
 
    # Starting PiVideoStream thread
    vs = PiVideoStream().start()

    # Camera Settings
    vs.camera.rotation = 180
    vs.camera.hflip = True
    vs.camera.vflip = False

    # Letting camera warmup 
    time.sleep(2.0)

    # Initializing button
    button_obj = button()

    # Variables used to get frames count
    frame_count = 0
    start_time = time.time()

    # Initialize image1 using image2 (only done first time)
    # We do this on order to analyze what the camera is actually seeing in different points of view (debugging mainly)
    image2 = vs.read()     
    image1 = image2
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # While loop used to keep getting the video stream
    still_scanning = True

    print 'Face recognition system ready. Press the button!'
    
    # Variable used for the while loop and controls the entire camera video recording
    running = True

    # List containing every person which the script greeted already
    people_found = []
    
    # Initializing variable used to know whether we found a face or not
    face_found = False
    
    while running:

        # Exceptions handler (used to handle errors regarding the button)
        try:

            # If button is pressed
            if button_obj.is_button_up():
                
                while still_scanning:
                    
                    # At this point the image is available as stream.array
                    # Getting frame from stream
                    image2 = vs.read()

                    # Getting start_time and frame amount everytime
                    start_time, frame_count = show_FPS(start_time, frame_count)
                    
                    # Convert to gray scale, which is easier to process things with
                    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    
                    # Get differences between the two greyed, blurred images
                    differenceimage = cv2.absdiff(gray_image1, gray_image2)
                    differenceimage = cv2.blur(differenceimage,(10,10)) 

                    # Get threshold of difference image based on THRESHOLD_SENSITIVITY variable
                    retval, thresholdimage = cv2.threshold( differenceimage, 25, 255, cv2.THRESH_BINARY )         
                    try:
                        thresholdimage, contours, hierarchy = cv2.findContours(thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) ############################       
                    except:       
                        contours, hierarchy = cv2.findContours(thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) #####################################
                        
                    # Save grayimage2 to grayimage1 ready for next image2
                    gray_image1 = gray_image2

                    # Showing what the camera is seeing from different points of view
                    #cv2.imshow('Difference Image',differenceimage) 
                    #cv2.imshow('OpenCV Threshold', thresholdimage)

                    # Creating detectMultiScale object which helps us retrive faces in images 
                    faces = haar_faces.detectMultiScale(

                                    gray_image2, 
                                    scaleFactor= 1.2, 
                                    minNeighbors= 4, 
                                    minSize= (30, 30), 
                                    flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # For every face found 
                    for (x, y, w, h) in faces:

                        # Crop and resize image to face
                        crop_image = resize(crop(gray_image2, x, y, w, h))

                        # Test face against model
                        label, confidence = model.predict(crop_image) ##########################################

                        # Getting statistics about face prediction
                        #print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(label, confidence)

                        # Retrieving from the database the name of the user where the id is the same as the returned label from the algorithm
                        cursor.execute('SELECT name FROM users WHERE id=?', (label,))

                        # Getting the username
                        user = cursor.fetchone()
                        username = user[0]

                        # Check if user was already greeted or not
                        already_done = False

                        # For all people found previously in the video check if they were already greeted
                        for users in people_found:
                            if username == users:
                                already_done = True


                        # Checking user identity
                        # If algorithm confidence level is less than 5000.0
                        # The more the confidence level is near 0, the more the user identity is confirmed
                        # The number with which the confidence level is compared needs to be changed frequently based on lighting, quality
                        # of the camera or other things. If we have many false negatives the number needs to be increased or viceversa
                        if confidence <  5000.0:

                            # Drawing green rectangle around face
                            cv2.rectangle(image2, (x,y), (x+w, y+h), (255,255,255), 1)

                            # Writing username above rectangle with an = (which indicates that we found the user true identity)
                            cv2.putText(image2, "ID=" + username, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                                            
                            # Displaying frames
                            cv2.imshow('Camera', image2)

                            # Append found people to the list
                            people_found.append(username)

                            # If user hasn't been greeted we greeet him
                            #if not already_done:

                                # Output greeting
                                #user_known_audio(username)
                                                
                        # If algorithm confidence level is between 5000.0 and 6000.0 
                        elif 5000.0 < confidence < 6000.0: 
                                                
                            # Drawing green rectangle around face
                            cv2.rectangle(image2, (x,y), (x+w, y+h), (255,255,255), 1) 

                            # Writing username above rectangle with an +- which indicates that the algorithm almost found the user identity
                            # +- can be displayed even when the algorithm finds a similar user to the one who stands in front of the camera
                            cv2.putText(image2, "ID+-" + username, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) ###################
                                            
                            # Displaying frames
                            cv2.imshow('Camera', image2)

                        # If algorithm confidence level is above or equal to 6000.0
                        # We didn't find the user identity
                        else:

                            # Drawing white rectangle around face
                            cv2.rectangle(image2, (x,y), (x+w, y+h), (255,255,255), 1) 
                                            
                            print 'Trying to recognize you...'
                                                
                            # Displaying frames
                            cv2.imshow('Camera', image2)
                                            
                    # Displaying frames even if faces are not found
                    cv2.imshow('Camera', image2)
                    key = cv2.waitKey(1) & 0xFF

                    # Close Window if q or button pressed
                    if button_obj.is_button_up() or key == ord('q'):

                        time.sleep(3.0)
                                        
                        # Destroying all windows and stopping thread
                        cv2.destroyAllWindows()

                        # Stopping camera thread
                        vs.stop()
                        print("Face detection ended")

                        # Stopping loops
                        still_scanning = False
                        running = False
                                                        
        # Handling KeyboardInterrupt error     
        except KeyboardInterrupt:
            button_obj.end()
    
#-----------------------------------------------------------------------------------------------
            
if __name__ == '__main__':
    
    # It's always preferable to perform database operations inside the try-catch clause in order to handle errors.
    try:
        
        # Declaring the variable model which contains our EigenFaceRecognizer object and stands at the base of the entire script
        model = cv2.face.createEigenFaceRecognizer() 

        # Getting possible moddel functions that can be used
        #help (model) 

        # Connecting to the database
        database = sqlite3.connect('dataset.db') 

        # With this attribute we can control what objects are returned for the TEXT data type.
        database.text_factory = str 

        # Declaring cursor in order to perform CRUD operations on the database
        cursor = database.cursor()

        # We ask to the user whether he wants to train the algorithm or not
        # We need to train it everytime the database or photos/users are updated
        answer = raw_input('Do you want to train the model ? If first time type y (y/n) ')

        # If yes train
        if answer is 'y':

            # We ask to the user whether he wants to update the database before training the model
            update_db = raw_input('Do you want to update the database ? (y/n) ')
                              
            # If yes update database                 
            if update_db is 'y':
                update_database(database, cursor)
                train(database, cursor, model)
            else:
                train(database, cursor, model)

        # Peform face recognition
        face_recognition(database, cursor, model)

    # Handling database errors
    except sqlite3.Error as e:
        raise e
        db.rollback()
        
    # Finally clause important because it closes the database connection everytime
    finally:
        database.close()
    

