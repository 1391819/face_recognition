#!/usr/bin/python
# -*- coding: utf8 -*-

# import the necessary packages
import io
import numpy as np
import RPi.GPIO as GPIO
import time
import cv2
import fnmatch
import sys
import subprocess
from random import randint
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import os
from gtts import gTTS
import datetime
import calendar

#-----------------------------------------------------------------------------------------------  
class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32, rotation=180, hflip=False, vflip=False):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.rotation = rotation
        self.camera.framerate = framerate
        self.camera.hflip = hflip
        self.camera.vflip = vflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
#-----------------------------------------------------------------------------------------------

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

    # If single digit minute, add 'oh' to the beginning
    if minute < 10:
        minute = "0" + str(minute)
    
    return str(hour) + " " + str(minute) + " " + ("PM" if pm else "AM")

#-----------------------------------------------------------------------------------------------
"""
def get_current_day_of_week():
    return calendar.day_name[ datetime.date.today().weekday() ]

#-----------------------------------------------------------------------------------------------

def get_current_day():
    return get_ordinal_string(datetime.datetime.now().day)

#-----------------------------------------------------------------------------------------------

def get_current_month():
    return calendar.month_name[ datetime.datetime.now().month ]

#-----------------------------------------------------------------------------------------------

def get_current_greet_time():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"
"""
#-----------------------------------------------------------------------------------------------
        
def download(message, outfile='message.mp3'):
    # Test: create an mp3
    message = gTTS(text=message, lang='it')
    message.save(outfile)

#-----------------------------------------------------------------------------------------------
        
class button(object):
	"""Class to represent the state and encapsulate access to the hardware of 
	the treasure box."""
	
	def end(self):
                GPIO.cleanup()
                
	def __init__(self):
		# Initialize button.
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(25, GPIO.IN)
		# Set initial box state.
		self.button_state = GPIO.input(25)

	def is_button_up(self):
		"""Return True when the box button has transitioned from down to up (i.e.
		the button was pressed)."""
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
			
def delete_duplicates(seq, idfun=None):
        # order preserving
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
            
#-----------------------------------------------------------------------------------------------
            
def get_random_number():

        number = randint(0, 5)

        return number
    
#-----------------------------------------------------------------------------------------------
    
haar_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

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
	return cv2.resize(image, 
					  (92, 112), 
					  interpolation=cv2.INTER_LANCZOS4)
    
#------------------------------------------------------------------------------------------------------------------
def show_FPS(start_time,frame_count):
    
    if frame_count >= 1000:
        duration = float(time.time() - start_time)
        FPS = float(frame_count / duration)
        print("Processing at %.2f fps last %i frames" %( FPS, frame_count))
        frame_count = 0
        start_time = time.time()
        
    else:
        frame_count += 1
        
    return start_time, frame_count

#-----------------------------------------------------------------------------------------------

def train():

        global model
        
	print "Reading training images..."
	faces = []
	labels = []
	people_count = 0
	
	# Read all positive images
	for filename in walk_files('./training/dataset', '*.pgm'):
		faces.append(prepare_image(filename))
		temp_file = os.path.basename(filename)
		temp_file = os.path.splitext(temp_file)[0]
		temp_file = temp_file[:-3]
		file = temp_file[:3]
		label = int(file)
		#print label
		labels.append(label)
		people_count += 1

		
	print 'Read', people_count, 'people.'

	# Train model
	print 'Training model...'
	#print faces
	#print labels
	model = cv2.face.createEigenFaceRecognizer()
	model.train(np.asarray(faces), np.asarray(labels))

	# Save model results
	model.save('training.xml')
	print 'Training data saved to training.xml!'
	
#-----------------------------------------------------------------------------------------------
	
def user_known_audio(username):
    
    greeting = "Buona giornata " + username + ", come stai ?"
    time = "Sono le " + get_current_time() + ". "

    download(greeting + time, outfile="audio_sample/greeting.mp3")
    subprocess.call("avplay -nodisp -autoexit audio_sample/greeting.mp3", shell=True)
    os.remove("audio_sample/greeting.mp3")

#-----------------------------------------------------------------------------------------------
            
def user_unknown_audio(list_unknown):

    num = get_random_number()
    #print num

    temp_audio_file = list_unknown[num]
    #print temp_audio_file

    audio_file = "audio_sample/" + temp_audio_file + ".mp3"

    subprocess.call("avplay -nodisp -autoexit " + audio_file, shell=True)
    
#-----------------------------------------------------------------------------------------------

def user_not_found_audio(list_not_found):

    num = get_random_number()
    #print num

    temp_audio_file = list_not_found[num]
    #print temp_audio_file

    audio_file = "audio_sample/" + temp_audio_file + ".mp3"

    subprocess.call("avplay -nodisp -autoexit " + audio_file, shell=True)

#-----------------------------------------------------------------------------------------------

def face_recognition():

    # Loading training data into the model
    print 'Loading training data...'
    model.load('training.xml')
    print 'Training data loaded!'

    # Initializing camera using threads
    print("Initializing Camera ....")
    
    vs = PiVideoStream().start()
    vs.camera.rotation = 180
    vs.camera.hflip = True
    vs.camera.vflip = False

    # Letting camera warmup 
    time.sleep(2.0)

    # Initializing button
    button_obj = button()

    # Auto cooldown for the camera
    t_start = 100
    t_end = 0

    # Lists containing people names
    temp_people = []
    people = []

    # Getting saved people names
    for filename in walk_files('./training/dataset', '*.pgm'):
        temp_file = os.path.basename(filename)
        temp_file = os.path.splitext(temp_file)[0]
        temp_file = temp_file[:-3]
        temp_people.append(temp_file)
        temp_people.sort()
        temp_people = delete_duplicates(temp_people)

    for person in temp_people:
        person_name = person[3:]
        people.append(person_name)

    # Getting audio samples
    list_unknown = ['unknown', 'unknown1', 'unknown2', 'unknown3', 'unknown4', 'unknown5']
    list_not_found = ['notfound', 'notfound1', 'notfound2', 'notfound3', 'notfound4', 'notfound5']
 
    frame_count = 0
    start_time = time.time()

    # initialize image1 using image2 (only done first time)
    image2 = vs.read()     
    image1 = image2
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    still_scanning = True

    user_unknown = False
    user_not_found = True

    cx = 0
    cy = 0
    cw = 0
    ch = 0
    
    print 'Face recognition system ready. Press the button!'

    running = True

    # Initializing variable used to know whether we found a face or not
    face_found = False
    
    while running:

        # Exceptions handler (mainly for the button
        try:

            if button_obj.is_button_up():
                
            
                # Do all stuff here
                # Until user presses q 
                while still_scanning:

                    image2 = vs.read()        
                    start_time, frame_count = show_FPS(start_time, frame_count)
                    
                    # initialize variables         
                    motion_found = False
                    biggest_area = 200
                    
                    # At this point the image is available as stream.array
                    # Convert to gray scale, which is easier
                    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    
                    # Get differences between the two greyed, blurred images
                    differenceimage = cv2.absdiff(gray_image1, gray_image2)
                    differenceimage = cv2.blur(differenceimage,(10,10))
                    
                    # Get threshold of difference image based on THRESHOLD_SENSITIVITY variable
                    retval, thresholdimage = cv2.threshold( differenceimage, 25, 255, cv2.THRESH_BINARY )         
                    try:
                        thresholdimage, contours, hierarchy = cv2.findContours(thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )        
                    except:       
                        contours, hierarchy = cv2.findContours(thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
                        
                    # Get total number of contours
                    total_contours = len(contours)
                    
                    # save grayimage2 to grayimage1 ready for next image2
                    gray_image1 = gray_image2
                    
                    # find contour with biggest area
                    for c in contours:
                        
                        # get area of next contour
                        found_area = cv2.contourArea(c)
                        
                        # find the middle of largest bounding rectangle
                        if found_area > biggest_area:
                            motion_found = True
                            biggest_area = found_area
                            (x, y, w, h) = cv2.boundingRect(c)
                            cx = int(x + w/2)   # put circle in middle of width
                            cy = int(y + h/6)   # put circle closer to top
                            cw = w
                            ch = h
                            
                    if motion_found:
                        # Do Something here with motion data

                        # Get last image/frame from stream
                        #image = vs.read()

                        # Convert image/frame to grayscale
                        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        

                        #cv2.rectangle(image2,(cx,cy),(x+cw,y+ch),(0,255,0), 1)                  

                        #print("Motion at cx=%3i cy=%3i  total_Contours=%2i  biggest_area:%3ix%3i=%5i" % (cx ,cy, total_contours, cw, ch, biggest_area))

                        #cv2.imshow('Difference Image',differenceimage)
                        cv2.imshow('OpenCV Threshold', thresholdimage)


                        # Get coordinates of single face in captured image/frame
                        result = detect_single(gray_image2)

                        # If the detection is not none we found a face else we didn't 
                        if result is not None:
                            face_found = True
                        else:
                            face_found = False


                        # Do stuff with the face
                        if face_found:
                            
                            # If we found a face -> Do stuff here
                            # Face coordinates
                            x, y, w, h = result

                            # Crop and resize image to face
                            crop_image = resize(crop(gray_image2, x, y, w, h))

                            # Test face against model
                            label, confidence = model.predict(crop_image)

                            # Getting statistics about face prediction
                            print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(label, confidence)

                            # Retrieving user info
                            username = people[label]
                        
                            # Draw rectangle arounf detected face
                            #cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,255), 1)

                            # Displaying frames
                            #cv2.imshow('Camera  (Press q in Window to Quit)', image)


                            # Checking user identity

                            if confidence < 3500.0:
                            
                                # Drawing green rectangle around face
                                cv2.rectangle(image2, (x,y), (x+w, y+h), (0,255,0), 1)

                                # Writing username above rectangle
                                cv2.putText(image2, username, (x, y - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
                            
                                # Displaying frames
                                cv2.imshow('Camera', image2)

                                # Write code if we want to stop everything once the user is detected
                                cv2.destroyAllWindows()
                                vs.stop()
                                print("Face detection ended")
                                still_scanning = False
                                running = False

                                # Output greeting and time
                                user_known_audio(username)
                            
                            else:
                                
                                # Drawing white rectangle around face
                                cv2.rectangle(image2, (x,y), (x+w, y+h), (255,255,255), 1)
                            
                                print 'Trying to recognize you...'
                         
                                if t_start == t_end:
                                    user_unknown_audio(list_unknown)
                                    still_scanning = False
                                    running = False
                                    
                                # Displaying frames
                                cv2.imshow('Camera', image2)
                                #continue
                        
                        else:
                            # If we didn't find a face -> Do stuff here
                             
                            if t_start == t_end:
                                user_not_found_audio(list_not_found)
                                still_scanning = False
                                running = False
            
                            # Displaying frames
                            cv2.imshow('Camera', image2)
                        
                        # Close Window if q pressed while movement status window selected
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            vs.stop()
                            print("Face detection ended")
                            still_scanning = False
                            running = False
                     
                        #-------------------------------------------------------------
                        
                    #-------------------------------------------------------------------
                        t_end += 1
            
        except KeyboardInterrupt:
            button_obj.end()
        
#-----------------------------------------------------------------------------------------------    
if __name__ == '__main__':
    train()
    #motion_track()
    face_recognition()
    

