# -*- coding: utf-8 -*-


# 4mB per image. frame rate 90. =>  Can put 11 seconds in 4Gb ram. 
from pypylon import pylon
import os
import cv2



def grab_images(num_images = 500, frame_rate = 90.0, exposure_time = 500,
                output_dir = r"c:\data\captured_images" , start_count = 0 ):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    
    # Set frame rate
    #if camera.AcquisitionFrameRateEnable.IsWritable():
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(frame_rate)
    
    # Optionally set exposure time
   # if camera.ExposureTimeAbs.IsWritable():
    camera.ExposureTime.SetValue(exposure_time)
    
    camera.MaxNumBuffer = 500 # 500 images in ram. ~ 2Gb
    
    # Start grabbing
    camera.StartGrabbingMax(num_images)
    
    # Image grabber
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    # frame_interval = 1.0 / frame_rate
    # last_time = time.time()
    
    i = start_count
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException) # 5000 // is time out
    
        if grabResult.GrabSucceeded():
            
            img = grabResult.Array  


    
            # Save image
            filename = os.path.join(output_dir, f"image_{i:03d}.png")
            # Save with OpenCV
            cv2.imwrite(filename, img)

            print(f"Saved {filename}")
            i += 1
    
            # # Ensure fixed frame rate
            # elapsed = time.time() - last_time
            # sleep_time = max(0, frame_interval - elapsed)
            # time.sleep(sleep_time)
            # last_time = time.time()
        else: 
            print("capture failed %i" % i)
    
        grabResult.Release()
    
    camera.StopGrabbing()
    camera.Close()
    print("Done capturing images.")