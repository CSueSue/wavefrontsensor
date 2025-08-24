import imageGrabber
import stageWrapper


if __name__ == "__main__":
    
    # connect to arduino
    stage = stageWrapper.arduino()
    stage.connect()
    
    
    # start stage move. dy = 50mm @ 10mm/s.
    stage.move(0,60.0, 10.0)
    
    # capture images. 
    imageGrabber.grab_images(num_images=60,frame_rate=10.0,output_dir=r'c:\data\test001')
    
    
    