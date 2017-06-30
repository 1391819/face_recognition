import cv2
import sys
import os

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

filename_temp = os.path.basename(imagePath)
filename = os.path.splitext(filename_temp)[0]

#print filename

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.13,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

idx = 0

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    idx+=1
    cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (255, 255, 255), 1)
    cropped = image[y-9:y+h+10, x-9:x+w+10]
    cropped = cv2.resize(cropped, (120, 120))
    grayCropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('faces/' + filename + '_face_' + str(idx) + '.png', grayCropped)
    
cv2.imshow("Faces found", image)
cv2.waitKey(0)
