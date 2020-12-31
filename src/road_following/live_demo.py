import cv2
import numpy as np
import time
from jetcam.csi_camera import CSICamera

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_steering_model_xy.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

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