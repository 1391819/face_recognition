from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.rotation = 180
camera.resolution = (640, 480)
camera.start_preview()
for i in range(5):
    sleep(3)
    camera.capture('/home/pi/Desktop/image%s.jpg' % i)
camera.stop_preview()
