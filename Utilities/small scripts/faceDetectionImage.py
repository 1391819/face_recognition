import cv2
import sys
import os

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

filename_temp = os.path.basename(imagePath)
filename = os.path.splitext(filename_temp)[0]

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
    minSize=(30, 30),
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.imwrite(filename + '_faces.png', image)
cv2.waitKey(0)
