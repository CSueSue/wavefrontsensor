import serial
import time

class arduino:
    def __init__(self):
        self.serial = None
        
    
    def connect(self):
        # connect with arduino
        self.serial = serial.Serial('COM11', 115200 , timeout =1 )
        time.sleep(1)
        

        
        
    def close(self):
        # close connection. 
        self.serial.close()
        
    
    def write(self, cmd):
        self.serial.write(cmd.encode()+'\n'.encode())
        

        
    
    def readline(self):
        if self.serial.in_waiting > 0:
            return self.serial.readline()
        else:
            return ''


    def homeX(self):
        self.write('2&')
        self.readline() # clear data 
        tstart = time.time()
        while time.time()-tstart < 120:
             s = self.readline()
             if len(s)>0:
                 print(s.decode())
                 break
    
    def homeY(self):
        self.write('3&')
        self.readline() # clear data 
        tstart = time.time()
        while time.time()-tstart < 120:
            s = self.readline()
            if len(s)>0:
                print(s.decode())
                break
        
    def move(self,dx,dy,speed, wait= True):
        # dx,dy in mm , speed in mm/s.
        self.readline() # clear data 
        if dx>0:
            dirx = 1
        else:
            dirx = 0
        if dy>0:
            diry = 1
        else:
            diry = 0
        
        self.write('1&'+str(abs(dx))+ '&'+str(abs(dy))+ '&' + str(dirx) + '&' + str(diry) + '&' + str(abs(speed))+ '&')
        tstart = time.time()
        while (time.time()-tstart < 120) & wait:
             s = self.readline()
             if len(s)>0:
                 print(s.decode())
                 break        