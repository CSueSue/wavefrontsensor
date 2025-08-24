import serial
import time

class arduino:
    def __init__(self):
        self.serial = None
        
    
    def connect(self):
        # connect with arduino
        self.serial = serial.Serial('COM8', 115200 , timeout =1 )
        time.sleep(1)
        

        
        
    def close(self):
        # close connection. 
        self.serial.close()
        
    
    def write(self, cmd):
        self.serial.write(cmd.encode()+'\n'.encode())
        

        
    
    def readline(self):
        if serial.in_waiting > 0:
            return self.serial.readline()
        else:
            return ''


    def homeX(self):
        self.write('2&')
 
    
    def homeY(self):
        self.write('3&')
        
    def move(self,dx,dy,speed):
        # dx,dy in mm , speed in mm/s.
        if dx>0:
            dirx = 1
        else:
            dirx = 0
        if dy>0:
            diry = 1
        else:
            diry = 0
        
        self.write('1&'+str(abs(dx))+ '&'+str(abs(dy))+ '&' + str(dirx) + '&' + str(diry) + '&' + str(abs(speed))+ '&')
        