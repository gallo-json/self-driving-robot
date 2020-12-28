import cv2
import numpy as np
import time
from jetcam.csi_camera import CSICamera

camera = CSICamera(width=224, height=224)

font = cv2.FONT_HERSHEY_PLAIN

prev_frame_time = 0
new_frame_time = 0

while True:
    frame = camera.read()
    
    new_frame_time = time.time()
    
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # puting the FPS count on the frame 

    cv2.putText(frame, fps, (10, 50), font, 2, (255, 0, 0), 3)

    cv2.imshow("Image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break