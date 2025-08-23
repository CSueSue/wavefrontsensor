# -*- coding: utf-8 -*-


from pypylon import pylon
import cv2
import time

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width


count = 1000
while count< 1100:#camera.IsGrabbing():

    numberOfImagesToGrab = 1
    camera.StartGrabbingMax(numberOfImagesToGrab)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        print("Gray value of first pixel: ", img[0, 0])
        
        cv2.imwrite("image_%i.tif" % count, img)
        time.sleep(0.2)
        count+=1

    grabResult.Release()
camera.Close()