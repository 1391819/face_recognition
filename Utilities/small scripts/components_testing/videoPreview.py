import picamera
from time import sleep

camera = picamera.PiCamera()

camera.rotation = 180
camera.hflip = True
camera.resolution = (800, 600)

frame = camera.start_preview()
sleep(10)
camera.stop_preview()
