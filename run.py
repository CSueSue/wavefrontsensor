import imageGrabber
import stageWrapper
from imageProcessing import analyzeDataset

if __name__ == "__main__":
    
    # connect to arduino
    stage = stageWrapper.arduino()
    stage.connect()
    
    # home stage
    stage.homeY()
    # start stage move. dy = 50mm @ 10mm/s.
    stage.move(0,130.0, 5.0)
    # home stage
    stage.homeX()
    # start stage move. dy = 50mm @ 10mm/s.
    stage.move(100.0,0.0, 5.0)
    
    delta_y = -60.0 #mm
    speed = 5.0 # mm/s
    framerate = 5.0 # Hz
    dy_im = speed/framerate  # mm 
    dx_im = -5.0
    N_x = 1
    N_im = int(round(abs(delta_y)/speed*framerate))
    
    for i in range(10):
        imdir = r'c:\data\repro_%i' %i
        for j in range(N_x):
        
            stage.move(0, delta_y, speed, wait = False)
            
            # capture images. 
            imageGrabber.grab_images(num_images=N_im,frame_rate=framerate,
                                      output_dir=imdir,exposure_time = 170, start_count = j*N_im)
        
            
    
            # stage.move(dx_im, 0, 1.0, wait = True)
            stage.move( 0 , -delta_y, speed , wait = True)
            
            
        
        # stage.move(dx_im*N_x*-1, 0, speed, wait = True)
        
        analyzeDataset(imdir, 0*-dx_im/1000, dy_im/1000, N_x, N_im,65)
        
    #stage.close()
    
    # # # connect to arduino
    # stage = stageWrapper.arduino()
    # stage.connect()
    
    # # home stage
    # stage.homeY()
    # # start stage move. dy = 50mm @ 10mm/s.
    # stage.move(0,100.0, 5.0)
    # # home stage
    # stage.homeX()
    # # start stage move. dy = 50mm @ 10mm/s.
    # stage.move(100.0,0.0, 5.0)
    
    # stage.close()