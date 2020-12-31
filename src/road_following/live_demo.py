import cv2
import numpy as np
import time
from jetcam.csi_camera import CSICamera

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

### LOAD MODEL

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

### Sliders

def empty(a): pass

cv2.namedWindow("Sliders")
cv2.resizeWindow("Sliders", 640, 240)

cv2.createTrackbar("speed gain", "Sliders", 0, 100, empty) 
cv2.createTrackbar("steering gain", "Sliders", 0, 100, empty) 
cv2.createTrackbar("steering kd", "Sliders", 0, 100, empty) 
cv2.createTrackbar("steering bias", "Sliders", -100, 100, empty) 

### INFERENCE

def infer(image, speed_gain, steering_gain, steering_kd, steering_bias):
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    return x, y

## CAMERA

camera = CSICamera(width=224, height=224)

font = cv2.FONT_HERSHEY_PLAIN

prev_frame_time = 0
new_frame_time = 0

while True:
    frame = camera.read()
    new_frame_time = time.time()

    speed_gain = cv2.getTrackbarPos("speed gain", "Sliders") / 100
    steering_gain = cv2.getTrackbarPos("steering gain", "Sliders") / 100
    steering_kd = cv2.getTrackbarPos("steering kd", "Sliders") / 100
    steering_bias = cv2.getTrackbarPos("steering bias", "Sliders") / 100

    x, y = infer(frame, speed_gain, steering_gain, steering_kd, steering_bias)

    cv2.circle(frame, (x, y), 8, (0, 255, 0), 3)

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = int(fps) 
    fps = str(fps) 
  
    cv2.putText(frame, fps, (10, 50), font, 2, (255, 0, 0), 3)

    cv2.imshow("Image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()